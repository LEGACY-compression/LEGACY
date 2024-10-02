import math
import torch
import torch.distributed as dist

# 
# Modified from https://github.com/sands-lab/layer-wise-aaai20
# 
def layerwise_compressed_comm(
    model, world_size, method, current_epoch):
    total_size=0
    for index, layer in enumerate(model.parameters()):
        flatten_grad = layer.grad.data.view(-1)

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
        elif method == 'Topk-policy':
            compress_grad = topk_policy(flatten_grad, current_epoch)

        elif method == "Thresholdv":
            compress_grad = threshold_compressor(flatten_grad)


        elif method == 'Exact':
            compress_grad = flatten_grad
        else:
            raise IOError(str(method)+ " compressor method not found")
        dist.all_reduce(compress_grad)

        if compress_grad.numel() > 0:
            compress_grad /= float(world_size)
            flatten_grad.copy_(compress_grad)
        else:
            flatten_grad.zero_()

group_splits = [10000]
group_compressions = [1,1]
number_epochs = 30
gradient_norm_memory=[]
current_iter=0
accordion_check_inter = 20
accordion_K = 1 #100

def set_compreesion_split_and_level(group_splits_, group_compressions_, number_epochs_, method=''):
    global group_compressions, group_splits, number_epochs
    if len(group_splits_) >0:
        group_splits = [int(i) for i in group_splits_.split(',')]
    if 'mix' in method:
        group_compressions = [  [float(i)/100 for i in phase_compressions_.split(',')] for phase_compressions_ in group_compressions.split(';')]
    else:
        group_compressions = [float(i)/100 for i in group_compressions_.split(',')]
    number_epochs = number_epochs_



def topk(flatten_grad, K):
    if K!=1:
        flatten_grad_abs = flatten_grad.abs()
        thres, _ = flatten_grad_abs.kthvalue(
            math.ceil(flatten_grad.numel() * (1 - K))
        )  # send >= thres
        flatten_grad[flatten_grad_abs < thres] = 0
    return flatten_grad

def randomk(flatten_grad, K):
    if K!=1:
        mask = torch.randperm(flatten_grad.numel(), device=flatten_grad.device).lt(
            flatten_grad.numel() * K
        )
        flatten_grad *= mask.float()
    return flatten_grad

def thresholdv(flatten_grad, V):
    flatten_grad_abs = flatten_grad.abs()
    flatten_grad[flatten_grad_abs < V] = 0
    return flatten_grad


#Layer size based compression
def layer_size_classification(flatten_grad):
    compression = group_compressions[-1]
    tensor_size = flatten_grad.numel()
    for ind, group_size in enumerate(group_splits):
        if tensor_size < group_size:
            compression = group_compressions[ind]
            break
    return compression

#Epoch/Iteration based compression
def iteration_calssification(current_epoch):
    compression = group_compressions[-1]
    epoch_portien= number_epochs/len(group_compressions)
    for ind in range(len(group_compressions)):
        if current_epoch < (ind+1)*epoch_portien:
            compression = group_compressions[ind]
            break
    return compression

#Layer size + Epoch based compression
def size_iteration_classification(current_epoch, flatten_grad):
    iteration_class = len(group_compressions)
    epoch_portien= number_epochs/len(group_compressions)
    for ind in range(len(group_compressions)):
        if current_epoch < (ind+1)*epoch_portien:
            iteration_class = ind
            break
    compression = group_compressions[iteration_class][-1]
    tensor_size = flatten_grad.numel()
    for ind, group_size in enumerate(group_splits):
        if tensor_size < group_size:
            compression = group_compressions[iteration_class][ind]
            break
    return compression

    

def topk_compression(flatten_grad):
    K=group_compressions[0] # use the first compression parameter uniformaly on all layers 
    return topk(flatten_grad, K)

def topk_layer(flatten_grad):
    K=layer_size_classification(flatten_grad) # decide compression parameter based on layer size
    return topk(flatten_grad, K)

def topk_adaptative(flatten_grad, current_epoch):
    K=iteration_calssification(current_epoch) # decide compression parameter based on training epoch
    return topk(flatten_grad, K)

def topk_policy(flaten_grad, current_epoch):
    K = size_iteration_classification(current_epoch, flaten_grad)  # use both layer size and current epoch to decide the compression parameter
    return topk(flaten_grad, K)

def randomk_compression(flatten_grad):
    K=group_compressions[0]
    return randomk(flatten_grad, K)

def randomk_layer(flatten_grad):
    K=layer_size_classification(flatten_grad)
    return randomk(flatten_grad, K)

def randomk_adaptative(flatten_grad, current_epoch):
    K= iteration_calssification(current_epoch)
    return randomk(flatten_grad, K)


def threshold_compressor(flatten_grad):
    V=group_compressions[0] * 100 #because we devide by 100 in the parsing
    return thresholdv(flatten_grad, V)


