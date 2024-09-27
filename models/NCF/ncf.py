# Copyright (c) 2018, deepakn94, codyaustun, robieta. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -----------------------------------------------------------------------
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.jit
# from apex.optimizers import FusedAdam
import os
import math
import time
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn

import utils
import dataloading
from neumf import NeuMF
from feature_spec import FeatureSpec
from neumf_constants import USER_CHANNEL_NAME, ITEM_CHANNEL_NAME, LABEL_CHANNEL_NAME
import json
import dllogger
from compressor import *
import torch.distributed as dist

def synchronized_timestamp():
    torch.cuda.synchronize()
    return time.time()

def get_world_size():
    """
    Gets total number of distributed workers or returns one if distributed is
    not initialized.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size


def all_reduce_item(value, op='sum'):
    """
    All-reduces single scalar value if distributed is in use
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if op == 'sum' or op == 'mean':
            dop = torch.distributed.ReduceOp.SUM
        elif op == 'min':
            dop = torch.distributed.ReduceOp.MIN
        elif op == 'max':
            dop = torch.distributed.ReduceOp.MAX
        elif op == 'product':
            dop = torch.distributed.ReduceOp.PRODUCT
        else:
            raise RuntimeError('Unsupported reduce op')

        backend = torch.distributed.get_backend()
        if backend == torch.distributed.Backend.NCCL:
            device = torch.device('cuda')
        elif backend == torch.distributed.Backend.GLOO:
            device = torch.device('cpu')
        else:
            raise RuntimeError('Unsupported distributed backend')

        tensor = torch.tensor(value, device=device)
        torch.distributed.all_reduce(tensor, dop)
        if op == 'mean':
            tensor /= get_world_size()
        ret = tensor.item()
    else:
        ret = value
    return ret

def parse_args():
    parser = ArgumentParser(description="Train a Neural Collaborative"
                                        " Filtering model")
    parser.add_argument('--data', type=str,
                        help='Path to the directory containing the feature specification yaml')
    parser.add_argument('--feature_spec_file', type=str, default='feature_spec.yaml',
                        help='Name of the feature specification file or path relative to the data directory.')
    parser.add_argument('-e', '--epochs', type=int, default=30,
                        help='Number of epochs for training')
    parser.add_argument('-b', '--batch_size', type=int, default=2 ** 20,
                        help='Number of examples for each iteration. This will be divided by the number of devices')
    parser.add_argument('--valid_batch_size', type=int, default=2 ** 20,
                        help='Number of examples in each validation chunk. This will be the maximum size of a batch '
                             'on each device.')
    parser.add_argument('-f', '--factors', type=int, default=64,
                        help='Number of predictive factors')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[256, 256, 128, 64],
                        help='Sizes of hidden layers for MLP')
    parser.add_argument('-n', '--negative_samples', type=int, default=4,
                        help='Number of negative examples per interaction')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.0045,
                        help='Learning rate for optimizer')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='Rank for test examples to be considered a hit')
    parser.add_argument('--seed', '-s', type=int, default=40,
                        help='Manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float, default=1.0,
                        help='Stop training early at threshold')
    parser.add_argument('--beta1', '-b1', type=float, default=0.25,
                        help='Beta1 for Adam')
    parser.add_argument('--beta2', '-b2', type=float, default=0.5,
                        help='Beta1 for Adam')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Epsilon for Adam')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability, if equal to 0 will not use dropout at all')
    parser.add_argument('--checkpoint_dir', default='', type=str,
                        help='Path to the directory storing the checkpoint file, '
                             'passing an empty path disables checkpoint saving')
    parser.add_argument('--load_checkpoint_path', default=None, type=str,
                        help='Path to the checkpoint file to be loaded before training/evaluation')
    parser.add_argument('--mode', choices=['train', 'test'], default='train', type=str,
                        help='Passing "test" will only run a single evaluation; '
                             'otherwise, full training will be performed')
    parser.add_argument('--grads_accumulated', default=1, type=int,
                        help='Number of gradients to accumulate before performing an optimization step')
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--log_path', default='log.json', type=str,
                        help='Path for the JSON training log')
    
    parser.add_argument('--method', type=str, default="Thresholdv")
    parser.add_argument('--log_dir', type=str, default='./all_logs/final2/')
    parser.add_argument('--memory', type=int, default=0)
    parser.add_argument('--group_splits', type=str, default='', help='threshold to create groups per layer size')
    parser.add_argument('--group_compressions', type=str, default='0.000001,0.1', help='compression ratio % to use per group')
    return parser.parse_args()


def init_distributed(args):
    args.world_size = int(os.environ.get('WORLD_SIZE', default=1))
    args.distributed = args.world_size > 1
    if args.distributed:
        # print("distributing")
        args.local_rank = int(os.environ['LOCAL_RANK'])
        print(args.local_rank)
        '''
        Set cuda device so everything is done on the right GPU.
        THIS MUST BE DONE AS SOON AS POSSIBLE.
        '''
        torch.cuda.set_device(args.local_rank)

        '''Initialize distributed communication'''
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
    else:
        args.local_rank = 0


def val_epoch(model, dataloader: dataloading.TestDataLoader, k, distributed=False, world_size=1):
    model.eval()
    user_feature_name = dataloader.channel_spec[USER_CHANNEL_NAME][0]
    item_feature_name = dataloader.channel_spec[ITEM_CHANNEL_NAME][0]
    label_feature_name = dataloader.channel_spec[LABEL_CHANNEL_NAME][0]
    with torch.no_grad():
        p = []
        labels_list = []
        losses = []
        for batch_dict in dataloader.get_epoch_data():
            user_batch = batch_dict[USER_CHANNEL_NAME][user_feature_name]
            item_batch = batch_dict[ITEM_CHANNEL_NAME][item_feature_name]
            label_batch = batch_dict[LABEL_CHANNEL_NAME][label_feature_name]
            prediction_batch = model(user_batch, item_batch, sigmoid=True).detach()

            loss_batch = torch.nn.functional.binary_cross_entropy(input=prediction_batch.reshape([-1]),
                                                                  target=label_batch)
            losses.append(loss_batch)

            p.append(prediction_batch)
            labels_list.append(label_batch)

        ignore_mask = dataloader.get_ignore_mask().view(-1, dataloader.samples_in_series)
        ratings = torch.cat(p).view(-1, dataloader.samples_in_series)
        ratings[ignore_mask] = -1
        labels = torch.cat(labels_list).view(-1, dataloader.samples_in_series)
        del p, labels_list

        top_indices = torch.topk(ratings, k)[1]

        # Positive items are always first in a given series
        labels_of_selected = torch.gather(labels, 1, top_indices)
        ifzero = (labels_of_selected == 1)
        hits = ifzero.sum()
        ndcg = (math.log(2) / (torch.nonzero(ifzero)[:, 1].view(-1).to(torch.float) + 2).log_()).sum()
        total_validation_loss = torch.mean(torch.stack(losses, dim=0))
        #  torch.nonzero may cause host-device synchronization

    if distributed:
        torch.distributed.all_reduce(hits, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(ndcg, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_validation_loss, op=torch.distributed.ReduceOp.SUM)
        total_validation_loss = total_validation_loss / world_size


    num_test_cases = dataloader.raw_dataset_length / dataloader.samples_in_series
    hr = hits.item() / num_test_cases
    ndcg = ndcg.item() / num_test_cases

    model.train()
    return hr, ndcg, total_validation_loss
communication=[0,0] #[communicated element, nb of iteration] => ratio = communication_element/(nbof_iter*model_size)
def get_compression_ratio(model_size):
    return(communication[0]/(communication[1]*model_size))

def get_communicated_layers():
    return communicated_layers
communicated_layers=[]
test_accs=[]
throughputs_per_ep=[]
train_time_per_ep = []
def layerwise_compressed_comm(
    model, world_size, method=None, current_epoch=0, anything=None
):
    global communication, communicated_layers
    # model_size = len(list(model.parameters()))

    communication[1]+=1
    communicated_layers.append([])
    total_size=0
    for index, layer in enumerate(model.parameters()):
        flatten_grad = layer.grad.data.view(-1)
        # if flatten_grad.numel()-torch.count_nonzero(flatten_grad).item()>1:
        #     print(flatten_grad.numel()-torch.count_nonzero(flatten_grad).item())
        total_size+=flatten_grad.numel()
        if method =='Randomk':
            compress_grad = randomk_compression(flatten_grad)
        elif method == "Randomk-layer":
            compress_grad = randomk_layer(flatten_grad)
        elif method == "Randomk-adaptative":
            compress_grad = randomk_adaptative(flatten_grad, current_epoch)

        elif method == 'Topk':
            compress_grad = topk_compression(flatten_grad)
        elif method == 'Topk-layer':
            compress_grad = topk_layer(flatten_grad)
        elif method == "Topk-adaptative":
            compress_grad = topk_adaptative(flatten_grad, current_epoch)

        elif method == "Thresholdv":
            compress_grad = threshold_compressor(flatten_grad)

        elif method == "Accordion-Topk":
            compress_grad = accordion_compressor(flatten_grad, index)

        elif method == 'Exact':
            compress_grad = flatten_grad
        else:
            raise IOError(str(method)+ " compressor method not found")
        non_zero_grad=torch.count_nonzero(compress_grad).item()
        # print(compress_grad.numel(), non_zero_grad)
        communication[0]= communication[0]+non_zero_grad
        communicated_layers[-1].append(non_zero_grad)
        dist.all_reduce(compress_grad)

        if compress_grad.numel() > 0:
            compress_grad /= float(world_size)
            flatten_grad.copy_(compress_grad)
        else:
            flatten_grad.zero_()
    # print(sum(communicated_layers[-1])/total_size)
# communication=[0,0] #[communicated element, nb of iteration] => ratio = communication_element/(nbof_iter*model_size)
# def get_compression_ratio(model_size):
#     return(communication[0]/(communication[1]*model_size))


# layer_communicated=[]
# def get_communicated():
#     return layer_communicated

# def layerwise_compressed_comm(
#     model,method=None, extras="", thresh_compress=None
# ):
#     global layer_communicated
#     layer_communicated.append([])
#     model_size = len(list(model.parameters()))
#     communication[1]+=1

#     for index, layer in enumerate(model.parameters()):
#         flatten_grad = layer.grad.data.view(-1)
#         # total_size+=flatten_grad.numel()
#         if method == "Randomk-layer":
#             flatten_grad = randomk_layer(flatten_grad, extras)
#         elif method == "Randomk-adaptative" and extras:
#             flatten_grad = randomk_adaptative(flatten_grad, extras)

#         elif method == 'Topk-layer':
#             flatten_grad = topk_layer(flatten_grad, extras)
#         elif method == "Topk-adaptative":
#             flatten_grad = topk_adaptative(flatten_grad, extras)
#         elif method == "Thresholdv" and thresh_compress:
#             # Threshold V
#             flatten_grad_abs = flatten_grad.abs()
#             # compress_grad = flatten_grad.clone()
#             flatten_grad[flatten_grad_abs < thresh_compress] = 0
#         else:
#             raise IOError("compress method not found")
#         non_zero_grad=torch.count_nonzero(flatten_grad).item()
#         layer_communicated[-1].append(non_zero_grad)
#         communication[0]= communication[0]+non_zero_grad
#         dist.all_reduce(flatten_grad)

#         if flatten_grad.numel() > 0:
#             flatten_grad /= float(torch.distributed.get_world_size())
#             # flatten_grad.copy_(compress_grad)
#         else:
#             flatten_grad.zero_()
#     return flatten_grad

def main():
    throughputs_per=[]
    time_loginter = []
    args = parse_args()
    init_distributed(args)
    world_size = torch.distributed.get_world_size()
    print("World size:", world_size)
    group_splits, group_compressions, epochs = args.group_splits, args.group_compressions, args.epochs

    set_compreesion_split_and_level(group_splits, group_compressions, epochs)

    memory= args.memory
    anything={ "memory": memory,'network': 'NCF', "gpu": 0, "compression_method": args.method,
               "group_splits": group_splits, "group_compressions": group_compressions, "epochs": epochs}
    

    accuracy_per_epoch=[]
    if args.local_rank == 0:
        dllogger.init(backends=[dllogger.JSONStreamBackend(verbosity=dllogger.Verbosity.VERBOSE,
                                                           filename=args.log_path),
                                dllogger.StdOutBackend(verbosity=dllogger.Verbosity.VERBOSE)])
    else:
        dllogger.init(backends=[])

    dllogger.metadata('train_throughput', {"name": 'train_throughput', 'unit': 'samples/s', 'format': ":.3e"})
    dllogger.metadata('best_train_throughput', {'unit': 'samples/s'})
    dllogger.metadata('mean_train_throughput', {'unit': 'samples/s'})
    dllogger.metadata('eval_throughput', {"name": 'eval_throughput', 'unit': 'samples/s', 'format': ":.3e"})
    dllogger.metadata('best_eval_throughput', {'unit': 'samples/s'})
    dllogger.metadata('mean_eval_throughput', {'unit': 'samples/s'})
    dllogger.metadata('train_epoch_time', {"name": 'train_epoch_time', 'unit': 's', 'format': ":.3f"})
    dllogger.metadata('validation_epoch_time', {"name": 'validation_epoch_time', 'unit': 's', 'format': ":.3f"})
    dllogger.metadata('time_to_target', {'unit': 's'})
    dllogger.metadata('time_to_best_model', {'unit': 's'})
    dllogger.metadata('hr@10', {"name": 'hr@10', 'unit': None, 'format': ":.5f"})
    dllogger.metadata('best_accuracy', {'unit': None})
    dllogger.metadata('best_epoch', {'unit': None})
    dllogger.metadata('validation_loss', {"name": 'validation_loss', 'unit': None, 'format': ":.5f"})
    dllogger.metadata('train_loss', {"name": 'train_loss', 'unit': None, 'format': ":.5f"})

    dllogger.log(data=vars(args), step='PARAMETER')

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if not os.path.exists(args.checkpoint_dir) and args.checkpoint_dir:
        print("Saving results to {}".format(args.checkpoint_dir))
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # sync workers before timing
    if args.distributed:
        torch.distributed.broadcast(torch.tensor([1], device="cuda"), 0)
    torch.cuda.synchronize()

    main_start_time = synchronized_timestamp()

    feature_spec_path = os.path.join(args.data, args.feature_spec_file)
    feature_spec = FeatureSpec.from_yaml(feature_spec_path)
    trainset = dataloading.TorchTensorDataset(feature_spec, mapping_name='train', args=args)
    testset = dataloading.TorchTensorDataset(feature_spec, mapping_name='test', args=args)
    train_loader = dataloading.TrainDataloader(trainset, args)
    test_loader = dataloading.TestDataLoader(testset, args)
    # make pytorch memory behavior more consistent later
    torch.cuda.empty_cache()

    # Create model
    user_feature_name = feature_spec.channel_spec[USER_CHANNEL_NAME][0]
    item_feature_name = feature_spec.channel_spec[ITEM_CHANNEL_NAME][0]
    label_feature_name = feature_spec.channel_spec[LABEL_CHANNEL_NAME][0]
    model = NeuMF(nb_users=feature_spec.feature_spec[user_feature_name]['cardinality'],
                  nb_items=feature_spec.feature_spec[item_feature_name]['cardinality'],
                  mf_dim=args.factors,
                  mlp_layer_sizes=args.layers,
                  dropout=args.dropout)

    # optimizer = FusedAdam(model.parameters(), lr=args.learning_rate,
    #                       betas=(args.beta1, args.beta2), eps=args.eps)

    optimizer =  torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                           betas=(args.beta1, args.beta2), eps=args.eps)
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # use torch.mean() with dim later to avoid copy to host
    # Move model and loss to GPU
    model = model.cuda()
    criterion = criterion.cuda()

    if args.distributed:
        model = (model)

    local_batch = args.batch_size // args.world_size
    traced_criterion = torch.jit.trace(criterion.forward,
                                       (torch.rand(local_batch, 1), torch.rand(local_batch, 1)))

    # print(model)
    # print("{} parameters".format(utils.count_parameters(model)))
    # small=0
    total_params=0
    l=[]
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_params = param.numel()
            l.append(layer_params)
    #         # if layer_params <10000:
    #         #     small+=layer_params
            total_params += layer_params
    #         print(f"Layer: {name}, Parameters: {layer_params}")
    # print(l)
    anything['model_layers']=l
    anything['total_params']=total_params
    if args.method == "Accordion-Topk":
        accordion_init_memory(len(l))
    # print(f"Total Parameters: {total_params}")
    # print('small',small,'large',total_params-small)
    # print('first',sum(l[:len(l)//2]),'last',sum(l[len(l)//2:]))
    # print('Warming up cudnn on random inputs on device', next(model.parameters()).device)
    # print('Warming up cudnn on random inputs on device', next(model.parameters()).device)
    if args.load_checkpoint_path:
        state_dict = torch.load(args.load_checkpoint_path)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    if args.mode == 'test':
        start = synchronized_timestamp()
        hr, ndcg, val_loss = val_epoch(model, test_loader, args.topk,
                                       distributed=args.distributed, world_size=args.world_size)
        val_time = synchronized_timestamp() - start
        eval_size = test_loader.raw_dataset_length
        eval_throughput = eval_size / val_time

        dllogger.log(step=tuple(), data={'best_eval_throughput': eval_throughput,
                                         'hr@10': hr,
                                         'validation_loss': float(val_loss.item())})
        return

    # this should always be overridden if hr>0.
    # It is theoretically possible for the hit rate to be zero in the first epoch, which would result in referring
    # to an uninitialized variable.
    max_hr = 0
    best_epoch = 0
    best_model_timestamp = synchronized_timestamp()
    train_throughputs, eval_throughputs = [], []
    train_time_per_ep = []
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for epoch in range(args.epochs):

        begin = synchronized_timestamp()
        batch_dict_list = train_loader.get_epoch_data()
        num_batches = len(batch_dict_list)
        for i in range(num_batches // args.grads_accumulated):
            for j in range(args.grads_accumulated):
                
                batch_idx = (args.grads_accumulated * i) + j
                batch_dict = batch_dict_list[batch_idx]

                user_features = batch_dict[USER_CHANNEL_NAME]
                item_features = batch_dict[ITEM_CHANNEL_NAME]

                user_batch = user_features[user_feature_name]
                item_batch = item_features[item_feature_name]

                label_features = batch_dict[LABEL_CHANNEL_NAME]
                label_batch = label_features[label_feature_name]

                with torch.cuda.amp.autocast(enabled=args.amp):
                    outputs = model(user_batch, item_batch)
                    loss = traced_criterion(outputs, label_batch.view(-1, 1))
                    loss = torch.mean(loss.float().view(-1), 0)

                scaler.scale(loss).backward()
                layerwise_compressed_comm(
                    model, world_size, args.method, current_epoch=epoch, anything=anything
                )
            # layerwise_compressed_comm(
            #         model, args.method, extras=(epoch, args.extras), thresh_compress=args.thresh_compress
            #     )
            scaler.step(optimizer)
            scaler.update()

            for p in model.parameters():
                p.grad = None

        del batch_dict_list
        train_time = synchronized_timestamp() - begin
        train_time = all_reduce_item(train_time, op='max')
        train_time_per_ep.append(train_time)
        begin = synchronized_timestamp()

        epoch_samples = train_loader.length_after_augmentation
        train_throughput = epoch_samples / train_time
        train_throughput = all_reduce_item(train_throughput, op='sum')
        train_throughputs.append(train_throughput)

        hr, ndcg, val_loss = val_epoch(model, test_loader, args.topk,
                                       distributed=args.distributed, world_size=args.world_size)
        accuracy_per_epoch.append(hr)
        val_time = synchronized_timestamp() - begin
        eval_size = test_loader.raw_dataset_length
        eval_throughput = eval_size / val_time
        eval_throughputs.append(eval_throughput)
        # abs_grad_infos['per_epoch_test_acc'].append(hr)
        # abs_grad_infos['per_epoch_test_loss'].append(float(val_loss.item()))
        if args.distributed:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss = loss / args.world_size

        dllogger.log(step=(epoch,),
                     data={'train_throughput': train_throughput,
                           'hr@10': hr,
                           'train_epoch_time': train_time,
                           'validation_epoch_time': val_time,
                           'eval_throughput': eval_throughput,
                           'validation_loss': float(val_loss.item()),
                           'train_loss': float(loss.item())})

        if hr > max_hr and args.local_rank == 0:
            max_hr = hr
            best_epoch = epoch
            print("New best hr!")
            if args.checkpoint_dir:
                save_checkpoint_path = os.path.join(args.checkpoint_dir, 'model.pth')
                print("Saving the model to: ", save_checkpoint_path)
                torch.save(model.state_dict(), save_checkpoint_path)
            best_model_timestamp = synchronized_timestamp()

        if args.threshold is not None:
            if hr >= args.threshold:
                print("Hit threshold of {}".format(args.threshold))
                break

    if args.local_rank == 0:

        dllogger.log(data={'best_train_throughput': max(train_throughputs),
                           'best_eval_throughput': max(eval_throughputs),
                           'mean_train_throughput': np.mean(train_throughputs),
                           'mean_eval_throughput': np.mean(eval_throughputs),
                           'best_accuracy': max_hr,
                           'best_epoch': best_epoch,
                           'time_to_target': synchronized_timestamp() - main_start_time,
                           'time_to_best_model': best_model_timestamp - main_start_time,
                           'validation_loss': float(val_loss.item()),
                           'train_loss': float(loss.item())},
                     step=tuple())
        compression_ratio=round(get_compression_ratio(total_params)*100,2)
        print("The compression ratio is :",str(round(compression_ratio,2)) )
        print('max hr', max_hr)

        d={"test_accuracy":accuracy_per_epoch, "layer_transmit":get_communicated_layers(), 'throughput_per_epouch':  train_throughputs,
            'time_per_epoch': train_time_per_ep, 'total_time': sum(train_time_per_ep)}
        d.update(anything)
  
        
        d["global_compression_ratio"] = compression_ratio
        save_path=args.log_dir
        compression_param = f"-split-thersh({anything['group_splits']}),group-ratio({anything['group_compressions']})"

        results_file='_'.join(['test'+str(int(args.seed)),anything['network'], str(args.method), 'epoch-'+str(anything['epochs']), compression_param,str(compression_ratio)+'-ratio',str(accuracy_per_epoch[-1])+'-acc', str(get_world_size())+'nodes'])
        results_file+='.json'
        with open(save_path+results_file, 'w') as json_file:
                    json.dump(d, json_file)


if __name__ == '__main__':
    main()
