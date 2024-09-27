#!/usr/bin/env python3

import datetime
import json
import os
import re
import time
import random
import numpy as np
import torch

import gradient_reducers
import tasks
from mean_accumulator import MeanAccumulator
from timer import Timer
import wandb
from opacus.grad_sample import GradSampleModule, GradSampleModuleExpandedWeights
from opacus.validators import ModuleValidator

"""
When you run this script, it uses the default parameters below.
To change them, you can make another script, say `experiment.py`
and write, e.g.
```
import train
train.config["num_epochs"] = 200
train.config["n_workers"] = 4
train.config["rank"] = 0
train.main()
```

The configuration overrides we used for all our experiments can be found in the folder schedule/neurips19.
"""

config = dict(
    average_reset_epoch_interval=30,
    distributed_backend="nccl",
    fix_conv_weight_norm=False,
    num_epochs=300,
    checkpoints=[],
    num_train_tracking_batches=1,
    optimizer_batch_size=128,  # per worker
    optimizer_conv_learning_rate=0.1,  # tuned for batch size 128
    optimizer_decay_at_epochs=[150, 250],
    optimizer_decay_with_factor=10.0,
    optimizer_learning_rate=0.1,  # Tuned for batch size 128x16
    optimizer_momentum_type="nesterov",
    optimizer_momentum=0.9,
    optimizer_reducer="ExactReducer",

    optimizer_memory=False,
    optimizer_reducer_compression=0.01,
    optimizer_reducer_beta=0.9,
    optimizer_reducer_alpha=0.5,
    optimizer_reducer_rank=4,
    optimizer_reducer_thresh=0.5,
    optimizer_reducer_rmsratio=0.8,
    optimizer_reducer_gamma=0.5,
    
    #optimizer_reducer_reuse_query=True,
    #optimizer_reducer_n_power_iterations=0,

    optimizer_scale_lr_with_factor=None,  # set to override world_size as a factor
    optimizer_scale_lr_with_warmup_epochs=5,  # scale lr by world size
    optimizer_mom_before_reduce=False,
    optimizer_wd_before_reduce=False,
    optimizer_weight_decay_conv=0.0001,
    optimizer_weight_decay_other=0.0001,
    optimizer_weight_decay_bn=0.0,
    task="Cifar",
    task_architecture="ResNet18",
    seed=42,
    rank=0,
    n_workers=1,
    distributed_init_file=None,
    log_verbosity=2,
    ### wandb related configs ###
    wandb_key=None,
    proj_name=None,
    run_name=None,
    entity=None,
    use_wandb=0,
    #############################
    accordion_k_low=0.1,
    accordion_k_high=0.99,
    accordion_detection_threshold=0.5,
    accordion_switch_freq=1,
    using_ibex=False,
    #### layer-size compression ###
    k_large=0.02,
    k_small = 0.2,
    ##### epoch compression #####
    k_last=0.9,
    k_first = 0.9,
    group_splits = '1',
    group_compressions = '1,0.1',
    ###### Variance Based Compression ######
    variance_alpha = 0.1,
    variance_ksi =0.999,
)
output_dir = "./output"  # will be overwritten by run.py

current_step = 0

def clean_up():
    torch.distributed.destroy_process_group()
    
def main():
    global current_step, output_dir
    print(output_dir)
    if config["rank"] == 0:
        print("Train settings:",config)
    else:
        time.sleep(0.1)
    # Update it to add wandb configs
    if config["use_wandb"]:
        if config["wandbkey"] != 'none':
            os.environ["WANDB_API_KEY"] = config["wandbkey"]
        if config['rank'] != 0:
            os.environ['WANDB_MODE'] = 'dryrun'  # all wandb.log are no-op

        wandb.init(project=config["proj_name"], name=config["run_name"], entity=config["entity"], config=config)
        print('wandb logging has been enabled ')

    os.environ['PYTHONHASHSEED']=str(config["seed"] + config["rank"])
    torch.manual_seed(config["seed"] + config["rank"])
    torch.cuda.manual_seed_all(config["seed"] + config["rank"])
    np.random.seed(config["seed"] + config["rank"])
    random.seed(config["seed"] + config["rank"])
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda:" +str(config["rank"]) if torch.cuda.is_available() else "cpu") 
    torch.cuda.set_device(device)
    print(output_dir)
    print("Device for training:", device)
    print ('Available devices:', torch.cuda.device_count())
    print ('Current cuda device:', torch.cuda.current_device())

    timer = Timer(verbosity_level=config["log_verbosity"], log_fn=metric)

    if torch.distributed.is_available():
        if config["distributed_init_file"] is None:
            config["distributed_init_file"] = os.path.join(output_dir, "dist_init")
        print(
            "Distributed init: rank {}/{} - {}".format(
                config["rank"], config["n_workers"], config["distributed_init_file"]
            )
        )
        torch.distributed.init_process_group(
            backend=config["distributed_backend"],
            init_method="file://" + os.path.abspath(config["distributed_init_file"]),
            timeout=datetime.timedelta(seconds=5),
            world_size=config["n_workers"],
            rank=config["rank"],
        )
    task = tasks.build(task_name=config["task"], device=device, timer=timer, **config)
    # errors = ModuleValidator.validate(task._model, strict=False)
    # task._model = ModuleValidator.fix(task._model)
    # # errr= ModuleValidator.validate(task._model, strict=False)
    # # print('err', errr)
    # task._model = GradSampleModuleExpandedWeights(task._model)
    reducer = get_reducer(device, timer, task.state)
    # print(task._model)
    eps = 1e-8
    bits_communicated = 0
    runavg_model = MeanAccumulator()
    memories = [torch.zeros_like(param) for param in task.state]
    momenta = [torch.empty_like(param) for param in task.state]  # need initialization
    send_buffers = [torch.zeros_like(param) for param in task.state]
    print(reducer)
    
    if isinstance(reducer, gradient_reducers.AccordionTopKReducer):
        len_params = 0
        for param in task.state:
            len_params += 1
        auto_scale_tensor = torch.zeros(len_params, device=device)
        prev_norms = [eps for param in task.state]
        curr_norms = [0 for param in task.state]
        prev_lrs = [0 for param in task.state]
    total_params = sum(param.numel() for param in task.state if param.requires_grad)
    model_layers = [param.numel() for param in task.state if param.requires_grad]
    global_bs = config["optimizer_batch_size"]*config["n_workers"]
    print("Total parameters to optimize:", total_params)
    print("Global batch Size:", global_bs)
    
    total_params_transmitted = 0
    test_accuracy=[]
    throughput_per_epoch=[]
    time_per_epoch = []
    num_samples=0
    current_time = time.time()
    for epoch in range(config["num_epochs"]):

        #print("Inside the epoch loop:", epoch)
        params_transmitted = 0
        epoch_metrics = MeanAccumulator()
        throughput_metrics = MeanAccumulator()
        training_time_metrics = MeanAccumulator()

        info({"state.progress": float(epoch) / config["num_epochs"], "state.current_epoch": epoch})

        # This seems fine ...
        # check_model_consistency_across_workers(task._model, epoch)

        # Determine per-parameter optimization parameters
        wds = [get_weight_decay(epoch, name) for name in task.parameter_names]

        # Reset running average of the model
        if epoch % config["average_reset_epoch_interval"] == 0:
            runavg_model.reset()

        train_loader = task.train_iterator(config["optimizer_batch_size"])
        batches_per_epoch = len(train_loader)
        for i, batch in enumerate(train_loader):
            #print("Inside the train loader at batch number:", i)
            current_step+=1
            num_samples+=len(batch.x)

            epoch_frac = epoch + i / len(train_loader)
            lrs = [get_learning_rate(epoch_frac, name) for name in task.parameter_names]

            with timer("batch", epoch_frac, verbosity=0):
                _, grads, metrics = task.batch_loss_and_gradient(batch)
                # print(epoch_frac, grads[0][0])
                # layer0= task.state[0]
                # print(layer0.grad_sample.mean(axis=0))
                # print('-----')
                epoch_metrics.add(metrics)

                # Compute some derived metrics from the raw gradients
                with timer("batch.reporting.lr", epoch_frac, verbosity=2):
                    whole_grad_sq_norm = 0
                    if np.random.rand() < 0.001:  # with a small probability
                        for name, param, grad, lr in zip(task.parameter_names, task.state, grads, lrs):
                            tags = {"weight": name.replace("module.", "")}
                            # metric(
                            #     "learning_rate",
                            #     {
                            #         "epoch": epoch_frac, 
                            #         "value": lr,
                            #     },
                            #     tags,
                            # )
                            # metric(
                            #     "effective_lr",
                            #     {
                            #         "epoch": epoch_frac,
                            #         "value": lr / max(l2norm(param).item() ** 2, 1e-8),
                            #     },
                            #     tags,
                            # )
                            tensor_norm = l2norm(grad)
                            # metric(
                            #     "grad_norm",
                            #     {"epoch": epoch_frac, "value": tensor_norm.item()},
                            #     tags,
                            # )
                            whole_grad_sq_norm += tensor_norm**2
                        whole_grad_norm = torch.sqrt(whole_grad_sq_norm)
                        tags = {"weight": "entire_model"}
                        # metric(
                        #     "grad_norm",
                        #     {"epoch": epoch_frac, "value": whole_grad_norm.item()},
                        #     tags,
                        # )
                            

                if config["optimizer_wd_before_reduce"]:
                    with timer("batch.weight_decay", epoch_frac, verbosity=2):
                        for grad, param, wd in zip(grads, task.state, wds):
                            if wd > 0:
                                grad.add_(param.detach(), alpha=wd)

                if config["optimizer_mom_before_reduce"]:
                    with timer("batch.momentum", epoch_frac, verbosity=2):
                        for grad, momentum in zip(grads, momenta):
                            if epoch == 0 and i == 0:
                                momentum.data = grad.clone().detach()
                            else:
                                if (
                                    config["optimizer_momentum_type"]
                                    == "exponential_moving_average"
                                ):
                                    momentum.mul_(config["optimizer_momentum"]).add_(
                                        grad, alpha=1 - config["optimizer_momentum"]
                                    )
                                else:
                                    momentum.mul_(config["optimizer_momentum"]).add_(grad)
                            replace_grad_by_momentum(grad, momentum)

                with timer("batch.accumulate", epoch_frac, verbosity=2):
                    for grad, memory, send_bfr in zip(grads, memories, send_buffers):
                        if config["optimizer_memory"]:
                            send_bfr.data[:] = grad + memory
                        else:
                            send_bfr.data[:] = grad

                with timer("batch.reduce", epoch_frac, verbosity=2):
                    if isinstance(reducer, gradient_reducers.AccordionTopKReducer):
                        cur_bits, cur_params = reducer.reduce(send_buffers, grads, memories, auto_scale_tensor, prev_norms, curr_norms, prev_lrs, lrs, epoch_frac)
                    elif isinstance(reducer, gradient_reducers.TopKEpochReducer):
                        cur_bits, cur_params = reducer.reduce(send_buffers, grads, memories, epoch)
                    elif isinstance(reducer, gradient_reducers.VarianceReducer):
                        cur_bits, cur_params = reducer.reduce(task.state, grads, memories)
                    else:
                        cur_bits, cur_params = reducer.reduce(send_buffers, grads, memories)
                    bits_communicated += cur_bits
                    params_transmitted += cur_params
                # if config["optimizer_memory"]:
                #     with timer("batch.reporting.compr_err", verbosity=2):
                #         model_memory_sq_norm = 0
                #         model_send_bfr_sq_norm = 0
                #         if np.random.rand() < 0.001:
                #             for name, memory, send_bfr in zip(
                #                 task.parameter_names, memories, send_buffers
                #             ):
                #                 tags = {"weight": name.replace("module.", "")}
                #                 memory_sq_norm = l2norm(memory)**2
                #                 send_bfr_sq_norm = l2norm(send_bfr)**2
                #                 rel_compression_error = memory_sq_norm / send_bfr_sq_norm
                #                 model_memory_sq_norm += memory_sq_norm
                #                 model_send_bfr_sq_norm += send_bfr_sq_norm
                                # metric(
                                #     "rel_compression_error",
                                #     {"epoch": epoch_frac, "value": rel_compression_error.item()},
                                #     tags,
                                # )
                            # tags = {"weight":"entire_model"}
                            # model_rel_compression_error = model_memory_sq_norm/model_send_bfr_sq_norm
                            # metric(
                            #         "rel_compression_error",
                            #         {"epoch": epoch_frac, "value": model_rel_compression_error.item()},
                            #         tags,
                            #     )

                if not config["optimizer_wd_before_reduce"]:
                    with timer("batch.wd", epoch_frac, verbosity=2):
                        for grad, param, wd in zip(grads, task.state, wds):
                            if wd > 0:
                                grad.add_(param.detach(), alpha=wd)

                if not config["optimizer_mom_before_reduce"]:
                    with timer("batch.mom", epoch_frac, verbosity=2):
                        for grad, momentum in zip(grads, momenta):
                            if epoch == 0 and i == 0:
                                momentum.data = grad.clone().detach()
                            else:
                                if (
                                    config["optimizer_momentum_type"]
                                    == "exponential_moving_average"
                                ):
                                    momentum.mul_(config["optimizer_momentum"]).add_(
                                        grad, alpha=1 - config["optimizer_momentum"]
                                    )
                                else:
                                    momentum.mul_(config["optimizer_momentum"]).add_(grad)
                            replace_grad_by_momentum(grad, momentum)

                with timer("batch.step", epoch_frac, verbosity=2):
                    for param, grad, lr in zip(task.state, grads, lrs):
                        param.data.add_(grad, alpha=-lr)
                        # print('999')

                if config["fix_conv_weight_norm"]:
                    with timer("batch.normfix", epoch_frac, verbosity=2):
                        for param_name, param in zip(task.parameter_names, task.state):
                            if is_conv_param(param_name):
                                param.data[:] /= l2norm(param)

                with timer("batch.update_runavg", epoch_frac, verbosity=2):
                    runavg_model.add(task.state_dict())

                # if config["optimizer_memory"]:
                #     with timer("batch.reporting.memory_norm", epoch_frac, verbosity=2):
                #         if np.random.rand() < 0.001:
                #             sum_of_sq = 0.0
                #             model_memory_inf_norm = 0.0
                #             for parameter_name, memory, lr in zip(task.parameter_names, memories, lrs):
                #                 tags = {"weight": parameter_name.replace("module.", "")}
                #                 sq_norm = torch.sum(memory ** 2)
                #                 memory_inf_norm = torch.max(memory.abs()).item()
                #                 model_memory_inf_norm = max(memory_inf_norm, model_memory_inf_norm)
                #                 sum_of_sq += sq_norm
                                # metric(
                                #     "memory_norm",
                                #     {"epoch": epoch_frac, "value": torch.sqrt(sq_norm).item()},
                                #     tags,
                                # )
                                # metric(
                                #     "memory_inf_norm",
                                #     {"epoch": epoch_frac, "value": memory_inf_norm },
                                #     tags,
                                # )
                            # tags = {"weight":"entire_model"}   
                            # metric(
                            #     "memory_norm",
                            #     {"epoch": epoch_frac, "value": torch.sqrt(sum_of_sq).item()},
                            #     tags,
                            # )
                            # metric(
                            #     "memory_inf_norm",
                            #     {"epoch": epoch_frac, "value": model_memory_inf_norm},
                            #     tags,
                            # )
                            
        total_params_transmitted += params_transmitted
        with timer("epoch_metrics.collect", epoch + 1.0, verbosity=2):
            density = params_transmitted/(batches_per_epoch*total_params)*100
            avg_density = total_params_transmitted/(batches_per_epoch*total_params*(epoch+1.0))*100
            epoch_metrics.reduce()

            if config["use_wandb"]:
                wandb.log({'density'+'/'+'current': density}, step=int(current_step))
                wandb.log({'density'+'/'+'average': avg_density}, step=int(current_step))
                wandb.log({'Epoch':epoch+1.0}, step=int(current_step))

            for key, value in epoch_metrics.value().items():
                metric(
                    key,
                    {"value": value.item(), "epoch": epoch + 1.0, "bits": bits_communicated, "density": density},
                    tags={"split": "train"},
                )
                metric(
                    f"last_{key}",
                    {"value": value.item(), "epoch": epoch + 1.0, "bits": bits_communicated, "density": density},
                    tags={"split": "train"},
                )

        with timer("test.last", epoch):
            task._zero_grad()
            test_stats = task.test()
            test_accuracy.append(test_stats["accuracy"].item())
            # print(type(test_stats["accuracy"]))
            for key, value in test_stats.items():
                # if key =="accuracy":
                #     test_accuracy.append[value.item()]
                metric(
                    f"last_{key}",
                    {"value": value.item(), "epoch": epoch + 1.0, "bits": bits_communicated, "density": density},
                    tags={"split": "test"},
                )

        with timer("test.runavg", epoch):
            test_stats = task.test(state_dict=runavg_model.value())
            for key, value in test_stats.items():
                metric(
                    f"runavg_{key}",
                    {"value": value.item(), "epoch": epoch + 1.0, "bits": bits_communicated, "density": density},
                    tags={"split": "test"},
                )

        if epoch in config["checkpoints"] and torch.distributed.get_rank() == 0:
            with timer("checkpointing"):
                save(
                    os.path.join(output_dir, "epoch_{:03d}".format(epoch)),
                    task.state_dict(),
                    epoch + 1.0,
                    test_stats,
                )
                # Save running average model @TODO
        
        
        elapsed = time.time()-current_time
        throughput = num_samples/elapsed
        throughput_metrics.add(throughput)
        throughput_metrics.reduce()
        training_time_metrics.add(elapsed)
        training_time_metrics.reduce()

        throughput_per_epoch.append(throughput_metrics.value()* config["n_workers"])
        time_per_epoch.append(training_time_metrics.value())
        current_time = time.time()
        num_samples = 0

        print(timer.summary())
        if config["rank"] == 0:
            print(f"time per epoch {time_per_epoch[-1]}")
            timer.save_summary(os.path.join(output_dir, "timer_summary.json"))

    info({"state.progress": 1.0})
    print(sum(time_per_epoch))
    if config["rank"] == 0:

        compression_param=""
        d={"test_accuracy":test_accuracy, "layer_transmit":gradient_reducers.get_communicated(), 'throughput_per_epouch': throughput_per_epoch,
            'time_per_epoch': time_per_epoch, 'total_time': sum(time_per_epoch), 'model_layers': model_layers, "total_params": total_params,
            "compression_method": config["optimizer_reducer"], "epochs": config["num_epochs"],
            'network': config["task_architecture"], "group_compressions": config["group_compressions"], "group_splits": config["group_splits"],
            "n_worker": config["n_workers"], 'memory': config["optimizer_memory"]}
        if (config["optimizer_reducer"] == "GlobalTopKReducer"
            or config["optimizer_reducer"] == "TopKReducer"):
            compression_param='-'+str(100*config["optimizer_reducer_compression"])
        elif config["optimizer_reducer"] == "GlobalCATReducer":
            compression_param= ''
        elif config["optimizer_reducer"] == "ThreshReducer":
            compression_param= '-'+str(100*config["optimizer_reducer_thresh"])

        elif config["optimizer_reducer"] == "AccordionTopKReducer":
            k_low=config["accordion_k_low"]
            k_high=config["accordion_k_high"]
            compression_param = f'-({100*k_low},{100*k_high})'
            d['k_low'] = k_low
            d['k_high'] = k_high
        elif config["optimizer_reducer"] == "VarianceReducer":
            alpha=config["variance_alpha"]
            ksi=config["variance_ksi"]
            compression_param = f'-({alpha},{ksi})'
            d['alpha'] = alpha
            d['ksi'] = ksi
        elif config["optimizer_reducer"] == "TopKLayerSizeReducer":
            group_compressions=config["group_compressions"]
            group_splits=config["group_splits"]
            compression_param = f'-split-thersh({group_splits}),group-ratio({group_compressions})'
        elif config["optimizer_reducer"] == "TopKEpochReducer":
            group_compressions=config["group_compressions"]
            group_splits=config["group_splits"]
            compression_param = f'group-ratio({group_compressions})'
        using_memory="without-EF"
        if config["optimizer_memory"]:
            using_memory="with-EF"
        
        if config['optimizer_reducer'] =="ExactReducer":
            compression_ratio=100
        else:
            all_transmit=np.array(d["layer_transmit"])
            print(all_transmit.shape)
            compression_ratio=(all_transmit.sum(axis=1).mean(axis=0))*100/total_params
        # print(compression_ratio)
        d["global_compression_ratio"] = compression_ratio


        data_set= 'Cifar10' if config['task']=='Cifar' else config['task']
        results_file="_".join(['test'+str(config["seed"]), data_set, config['task_architecture'], config['optimizer_reducer']+ compression_param, str(config["num_epochs"])+'ep', str(round(compression_ratio,2))+'%-ratio',
                                str(config["n_workers"])+'gpus', using_memory, 'seed'+str(config["seed"]), str(round(test_accuracy[-1]*100))+'acc'])+'.json'
        print(results_file)
        with open(os.path.join(os.path.expanduser(output_dir), results_file), 'w') as json_file:
                    json.dump(d, json_file)
    clean_up()



def save(destination_path, model_state, epoch, test_stats):
    """Save a checkpoint to disk"""
    # Workaround for RuntimeError('Unknown Error -1')
    # https://github.com/pytorch/pytorch/issues/10577
    time.sleep(1)

    torch.save(
        {"epoch": epoch, "test_stats": test_stats, "model_state_dict": model_state},
        destination_path,
    )


def get_weight_decay(epoch, parameter_name):
    """Take care of differences between weight decay for parameters"""
    if is_conv_param(parameter_name):
        return config["optimizer_weight_decay_conv"]
    elif is_batchnorm_param(parameter_name):
        return config["optimizer_weight_decay_bn"]
    else:
        return config["optimizer_weight_decay_other"]


def get_learning_rate(epoch, parameter_name):
    """Apply any learning rate schedule"""
    if is_conv_param(parameter_name):
        lr = config["optimizer_conv_learning_rate"]
    else:
        lr = config["optimizer_learning_rate"]

    if config["optimizer_scale_lr_with_warmup_epochs"]:
        warmup_epochs = config["optimizer_scale_lr_with_warmup_epochs"]
        max_factor = config.get("optimizer_scale_lr_with_factor", None)
        if max_factor is None:
            max_factor = (
                torch.distributed.get_world_size() if torch.distributed.is_available() else 1.0
            )
        factor = 1.0 + (max_factor - 1.0) * min(epoch / warmup_epochs, 1.0)
        lr *= factor

    for decay_epoch in config["optimizer_decay_at_epochs"]:
        if epoch >= decay_epoch:
            lr /= config["optimizer_decay_with_factor"]
        else:
            return lr
    return lr


def is_conv_param(parameter_name):
    """
    Says whether this parameter is a conv linear layer that 
    needs a different treatment from the other weights
    """
    return "conv" in parameter_name and "weight" in parameter_name


def is_batchnorm_param(parameter_name):
    """
    Is this parameter part of a batchnorm parameter?
    """
    return re.match(r""".*\.bn\d+\.(weight|bias)""", parameter_name)


def replace_grad_by_momentum(grad, momentum):
    """
    Inplace operation that applies momentum to a gradient.
    This distinguishes between types of momentum (heavy-ball vs nesterov)
    """
    if config["optimizer_momentum_type"] == "heavy-ball":
        grad.data[:] = momentum
    if config["optimizer_momentum_type"] == "exponential_moving_average":
        grad.data[:] = momentum
    elif config["optimizer_momentum_type"] == "nesterov":
        grad.data[:] += momentum
    else:
        raise ValueError("Unknown momentum type")


def get_reducer(device, timer, parameters):
    """Configure the reducer from the config"""
    if (
        config["optimizer_reducer"] == "GlobalTopKReducer"
        or config["optimizer_reducer"] == "TopKReducer"
    ):
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            compression=config["optimizer_reducer_compression"],
        )
    elif config["optimizer_reducer"] == "GlobalCATReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
        )
    elif config["optimizer_reducer"] == "ThreshReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            thresh=config["optimizer_reducer_thresh"]
        )
    elif config["optimizer_reducer"] == "AccordionTopKReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            k_low=config["accordion_k_low"],
            k_high=config["accordion_k_high"],
            detection_threshold=config["accordion_detection_threshold"],
            switch_freq=config["accordion_switch_freq"]
        )
    elif config["optimizer_reducer"] == "TopKLayerSizeReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            group_splits=config["group_splits"],
            group_compressions=config["group_compressions"],
        )
    elif config["optimizer_reducer"] == "TopKEpochReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            group_compressions=config["group_compressions"],
            num_epochs=config["num_epochs"]
        )
    elif config["optimizer_reducer"] == "VarianceReducer":
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            random_seed=config["seed"],
            device=device,
            timer=timer,
            parameters = parameters,
            var_alpha=config["variance_alpha"],
            var_ksi=config["variance_ksi"]
        )
    else:
        return getattr(gradient_reducers, config["optimizer_reducer"])(
            config["seed"], device, timer
        )


@torch.jit.script
def l2norm(tensor):
    """Compute the L2 Norm of a tensor in a fast and correct way"""
    # tensor.norm(p=2) is buggy in Torch 1.0.0
    # tensor.norm(p=2) is really slow in Torch 1.0.1
    return torch.sqrt(torch.sum(tensor ** 2))


def log_info(info_dict):
    """Add any information to MongoDB
       This function will be overwritten when called through run.py"""
    pass


def log_metric(name, values, tags={}):
    """Log timeseries data
       This function will be overwritten when called through run.py"""
    global current_step
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{key}:{value:7.3f}")

    #Wandb logging
        if config["use_wandb"]:
            for _, tag in tags.items():
                wandb.log({str(name)+'/'+str(tag)+'_'+str(key): value}, step=int(current_step))

    values = ", ".join(value_list)
    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)
    print("{name:30s} - {values} ({tags})".format(name=name, values=values, tags=tags))


def info(*args, **kwargs):
    if config["rank"] == 0:
        log_info(*args, **kwargs)


def metric(*args, **kwargs):
    if config["rank"] == 0:
        log_metric(*args, **kwargs)


def check_model_consistency_across_workers(model, epoch):
    signature = []
    for name, param in model.named_parameters():
        signature.append(param.view(-1)[0].item())

    rank = config["rank"]
    signature = ",".join(f"{x:.4f}" for x in signature)
    print(f"Model signature for epoch {epoch:04d} / worker {rank:03d}:\n{signature}")


if __name__ == "__main__":
    print(output_dir)
    main()
