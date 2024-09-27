from inspect import signature
from collections import defaultdict, namedtuple
import json
import os
import time
import numpy as np
from functools import singledispatch
import torch.distributed as dist
import torch
from torch import nn
import math
from compressor import *
#####################
# utilsy
#####################
torch.manual_seed(42)
torch.cuda.manual_seed(42)

class Timer:
    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.time()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.synch()
        self.times.append(time.time())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t


localtime = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class TableLogger:
    def append(self, output):
        if not hasattr(self, "keys"):
            self.keys = output.keys()
            print(*(f"{k:>12s}" for k in self.keys))
        filtered = [output[k] for k in self.keys]
        print(*(f"{v:12.4f}" if isinstance(v, float) else f"{v:12}" for v in filtered))


#####################
## data preprocessing
#####################

cifar10_mean = (
    0.4914,
    0.4822,
    0.4465,
)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (
    0.2471,
    0.2435,
    0.2616,
)  # equals np.std(train_set.train_data, axis=(0,1,2))/255


def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def pad(x, border=4):
    return np.pad(
        x, [(0, 0), (border, border), (border, border), (0, 0)], mode="reflect"
    )


def transpose(x, source="NHWC", target="NCHW"):
    return x.transpose([source.index(d) for d in target])


#####################
## data augmentation
#####################


class Crop(namedtuple("Crop", ("h", "w"))):
    def __call__(self, x, x0, y0):
        return x[:, y0 : y0 + self.h, x0 : x0 + self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {"x0": range(W + 1 - self.w), "y0": range(H + 1 - self.h)}

    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)


class FlipLR(namedtuple("FlipLR", ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x

    def options(self, x_shape):
        return {"choice": [True, False]}


class Cutout(namedtuple("Cutout", ("h", "w"))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:, y0 : y0 + self.h, x0 : x0 + self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {"x0": range(W + 1 - self.w), "y0": range(H + 1 - self.h)}


class Transform:
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k, v) in choices.items()}
            data = f(data, **args)
        return data, labels

    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, "output_shape") else x_shape
            self.choices.append(
                {k: np.random.choice(v, size=N) for (k, v) in options.items()}
            )


#####################
## dict utils
#####################

union = lambda *dicts: {k: v for d in dicts for (k, v) in d.items()}


def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict):
            yield from path_iter(val, (*pfx, name))
        else:
            yield ((*pfx, name), val)


#####################
## graph building
#####################

sep = "_"
RelativePath = namedtuple("RelativePath", ("parts"))
rel_path = lambda *parts: RelativePath(parts)


def build_graph(net):
    net = dict(path_iter(net))
    default_inputs = [[("input",)]] + [[k] for k in net.keys()]
    with_default_inputs = lambda vals: (
        val if isinstance(val, tuple) else (val, default_inputs[idx])
        for idx, val in enumerate(vals)
    )
    parts = (
        lambda path, pfx: tuple(pfx) + path.parts
        if isinstance(path, RelativePath)
        else (path,)
        if isinstance(path, str)
        else path
    )
    return {
        sep.join((*pfx, name)): (val, [sep.join(parts(x, pfx)) for x in inputs])
        for (*pfx, name), (val, inputs) in zip(
            net.keys(), with_default_inputs(net.values())
        )
    }


#####################
## training utils
#####################


@singledispatch
def cat(*xs):
    raise NotImplementedError


@singledispatch
def to_numpy(x):
    raise NotImplementedError


class PiecewiseLinear(namedtuple("PiecewiseLinear", ("knots", "vals"))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]
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

class StatsLogger:
    def __init__(self, keys):
        self._stats = {k: [] for k in keys}

    def append(self, output):
        for k, v in self._stats.items():
            v.append(output[k].detach())

    def stats(self, key):
        return cat(*self._stats[key])

    def mean(self, key):
        return np.mean(to_numpy(self.stats(key)), dtype=float)

communication=[0,0] #[communicated element, nb of iteration] => ratio = communication_element/(nbof_iter*model_size)
def get_compression_ratio(model_size):
    return(communication[0]/(communication[1]*model_size))
def get_compression_ratio2():
    return(sum(communication)-communication[0]-communication[1])/(len(communication)-2)
def get_communicated_layers():
    return communicated_layers
communicated_layers=[]
test_accs=[]
throughputs_per_ep=[]
train_time_per_ep = []
def layerwise_compressed_comm(
    model, world_size, method=None, K=None, V=None, qstates=None, current_epoch="", anything=None
):
    global communication, communicated_layers
    # model_size = len(list(model.parameters()))
    communication[1]+=1
    communicated_layers.append([])
    iter_volume=0
    total_size=0
    for index, layer in enumerate(model.parameters()):
        if method == 'Variance_based':
            flatten_grad = layer.grad_sample
        else:
            flatten_grad = layer.grad.data.view(-1)
        if anything['memory']== 1 and method != "Adacomp":
            flatten_grad = error_feedback_compensate(flatten_grad, index)
        total_size+=layer.grad.data.view(-1).numel()
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

        elif method == "Accordion-Topk":
            compress_grad = accordion_compressor(flatten_grad, index)

        elif method == "Adacomp":
            compress_grad = adacomp_layer(flatten_grad, index)
        
        elif method == 'Variance_based':
            start=time.time()
            compress_grad = variance_based(flatten_grad, index)
            print(f'{index} layer need {time.time()-start}')
        elif method == 'Exact':
            compress_grad = flatten_grad
        else:
            raise IOError(str(method)+ " compressor method not found")
        

        if anything['memory']== 1 and method != "Adacomp":
            error_feedback_update(compress_grad, index)
        non_zero_grad=torch.count_nonzero(compress_grad).item()
        # print(compress_grad.numel(), non_zero_grad)
        communication[0]= communication[0]+non_zero_grad
        communicated_layers[-1].append(non_zero_grad)
        dist.all_reduce(compress_grad)

        if compress_grad.numel() > 0:
            compress_grad /= float(world_size)
            layer.grad.data.view(-1).copy_(compress_grad)
        else:
            layer.grad.data.view(-1).zero_()
    communication.append(iter_volume/total_size)


def entiremodel_compressed_comm(
    model, world_size, method=None, K=None, V=None, qstates=None, current_epoch="", anything=None
):
    global communication, communicated_layers
    # model_size = len(list(model.parameters()))
    communication[1]+=1
    communicated_layers.append([])
    iter_volume=0
    total_size=0
    # concat model grads into one flattened vector
    vec = []
    for param in model.parameters():
        # Ensure the parameters are located in the same device
        # param_device = _check_param_device(param, param_device)

        vec.append(param.grad.data.view(-1))

    flatten_grad = torch.cat(vec)

    total_size=flatten_grad.numel()
    if method =='Adacomp':
        compress_grad = adacomp_compression(flatten_grad, anything['bins'])


    non_zero_grad=torch.count_nonzero(compress_grad).item()
    # print(compress_grad.numel(), non_zero_grad)
    communication[0]= communication[0]+non_zero_grad

    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        layer_non_zero = torch.count_nonzero(compress_grad[pointer : pointer + num_param]).item()
       
        # Increment the pointer
        pointer += num_param

        communicated_layers[-1].append(layer_non_zero)
    # print(compress_grad.numel(), non_zero_grad, sum(communicated_layers[-1]), communicated_layers[-1])

    # Perform All reduce (sum)
    dist.all_reduce(compress_grad)

    # average gradients
    if compress_grad.numel() > 0:
        compress_grad /= float(world_size)
        flatten_grad.copy_(compress_grad)
    else:
        flatten_grad.zero_()

    # Restore the gradient data into each indiviual layer of the model
    # Flag for the device where the parameter is located
    param_device = None
    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in model.parameters():
        # Ensure the parameters are located in the same device
        # param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.grad.data = (
            flatten_grad[pointer : pointer + num_param].view_as(param).data
        )
        # Increment the pointer
        pointer += num_param

criterion = nn.CrossEntropyLoss(reduction='none')

def run_batches(
    model,
    batches,
    training,
    world_size,
    optimizer_step=None,
    stats=None,
    compress=None,
    method=None,
    K=None,
    V=None,
    qstates=None,
    epoch=None,
    current_epoch="",
    anything=None
):
    stats = stats or StatsLogger(("loss", "correct"))
    model.train(training)
    # for X, labels in batches:
    #     X=X.to(device)
    #     labels= labels.to(device)
    #     output = model(X)
    #     loss = criterion(output, labels)
    #     pred = output.data.max(1, keepdim=True)[1]
    #     correct = pred.eq(labels.data.view_as(pred))
    #     resul={'loss' : loss, 'correct': correct}
    #     stats.append(resul)
    #     if training:
    #         loss.sum().backward()
    for batch in batches:        
        output = model(batch)
        stats.append(output)
        if training:
            output["loss"].sum().backward()
            
            if compress == "layerwise":
                layerwise_compressed_comm(
                    model, world_size, method, K, V, qstates, current_epoch=current_epoch, anything=anything
                )
            elif compress == "entiremodel":
                entiremodel_compressed_comm(
                    model, world_size, method, K, V, qstates, current_epoch=current_epoch, anything=anything
                )
            else:
                # no compression, communicate the gradients layer-by-layer
                for layer in model.parameters():
                    dist.all_reduce(layer.grad.data)
                    layer.grad.data /= world_size
            # previous_gradients = torch.cat([prev_grad.grad
            optimizer_step()
            model.zero_grad()
    return stats


def train_epoch(
    model,
    train_batches,
    test_batches,
    optimizer_step,
    timer,
    world_size,
    test_time_in_total=True,
    compress=None,
    method=None,
    K=None,
    V=None,
    qstates=None,
    current_epoch="",
    anything=None
):
    train_stats, train_time = (
        run_batches(
            model,
            train_batches,
            True,
            world_size,
            optimizer_step,
            compress=compress,
            method=method,
            K=K,
            V=V,
            qstates=qstates,
            current_epoch=current_epoch,
            anything=anything
        ),
        timer(),
    )
    test_stats, test_time = run_batches(
        model, test_batches, False, world_size, current_epoch=current_epoch, anything=anything
    ), timer(test_time_in_total)
    # abs_grad_infos['per_epoch_test_acc'].append(test_stats.mean("correct"))
    # abs_grad_infos['per_epoch_test_loss'].append(test_stats.mean("loss"))
    return {
        "train_time": train_time,
        "train_loss": train_stats.mean("loss"),
        "train_acc": train_stats.mean("correct"),
        "test_time": test_time,
        "test_loss": test_stats.mean("loss"),
        "test_acc": test_stats.mean("correct"),
        "total_time": timer.total_time,
    }


def get_test_nb(file_suffix, json_dir='json_output'):
    test=0
    for file in os.listdir(json_dir):
        if file_suffix in file:
            test+=1
    return str(test)

device = None
def train(
    model,
    optimizer,
    train_batches,
    test_batches,
    epochs,
    master_address,
    world_size,
    rank,
    loggers=(),
    test_time_in_total=True,
    timer=None,
    compress=None,
    method=None,
    K=None,
    V=None,
    qstates=None,
    anything=None
):
    global device, test_accs, throughputs_per_ep, train_time_per_ep

    device = anything['device']
    dist.init_process_group(
        backend="gloo", init_method=master_address, world_size=world_size, rank=rank
    )
    timer = timer or Timer()
    for epoch in range(epochs):
        epoch_stats = train_epoch(
            model,
            train_batches,
            test_batches,
            optimizer.step,
            timer,
            world_size,
            test_time_in_total=test_time_in_total,
            compress=compress,
            method=method,
            K=K,
            V=V,
            qstates=qstates,
            current_epoch=epoch,
            anything=anything
        )
   
        per_epoch_throughput= len(train_batches.dataset)/epoch_stats["train_time"]
        per_epoch_throughput= all_reduce_item(per_epoch_throughput, op='sum')
        per_epoch_time= all_reduce_item(epoch_stats["train_time"], op='sum')
        test_accs.append(epoch_stats["test_acc"])
        throughputs_per_ep.append(per_epoch_throughput)
        train_time_per_ep.append(per_epoch_time)  
        # anything["lr_scheduler"].step()
        anything['optimizer'].sync_param_updates()
        summary = union(
            {
                "epoch": epoch + 1,
                "lr": optimizer.param_values()["lr"] * train_batches.batch_size,
            },
            epoch_stats,
        )
        for logger in loggers:
            logger.append(summary)
            
    if rank ==0:
        d={"version": '1.0', "test_accuracy":test_accs, "layer_transmit":communicated_layers, 'throughput_per_epouch': throughputs_per_ep,
            'time_per_epoch': train_time_per_ep, 'total_time': sum(train_time_per_ep), "n_worker": get_world_size(), 'compression_type': compress}
        d.update(anything)
        del d['optimizer']
        del d['device']
        d['compression_method'] = method
        compression_ratio=round(get_compression_ratio(anything['total_params'])*100,3)
        d["global_compression_ratio"] = compression_ratio
        save_path='./all_logs/variance_tests/'
        compression_param = f"-split-thersh({anything['group_splits']}),group-ratio({anything['group_compressions']})"

        results_file='_'.join(['test'+str(int(K)), 'Cifar10', anything['network'], str(method)+compression_param, 'epoch-'+str(anything['epochs'])
                            ,str(compression_ratio)+'-ratio',str(test_accs[-1])+'-acc', str(get_world_size())+'nodes'])
        results_file+='.json'
        with open(save_path+results_file, 'w') as json_file:
                    json.dump(d, json_file)
    return summary


#####################
## network visualisation (requires pydot)
#####################
class ColorMap(dict):
    palette = (
        "bebada,ffffb3,fb8072,8dd3c7,80b1d3,fdb462,b3de69,fccde5,bc80bd,ccebc5,ffed6f,1f78b4,33a02c,e31a1c,ff7f00,"
        "4dddf8,e66493,b07b87,4e90e3,dea05e,d0c281,f0e189,e9e8b1,e0eb71,bbd2a4,6ed641,57eb9c,3ca4d4,92d5e7,b15928"
    ).split(",")

    def __missing__(self, key):
        self[key] = self.palette[len(self) % len(self.palette)]
        return self[key]


def make_pydot(nodes, edges, direction="LR", sep=sep, **kwargs):
    import pydot

    parent = lambda path: path[:-1]
    stub = lambda path: path[-1]

    class Subgraphs(dict):
        def __missing__(self, path):
            subgraph = pydot.Cluster(
                sep.join(path),
                label=stub(path),
                style="rounded, filled",
                fillcolor="#77777744",
            )
            self[parent(path)].add_subgraph(subgraph)
            return subgraph

    subgraphs = Subgraphs()
    subgraphs[()] = g = pydot.Dot(rankdir=direction, directed=True, **kwargs)
    g.set_node_defaults(shape="box", style="rounded, filled", fillcolor="#ffffff")
    for node, attr in nodes:
        path = tuple(node.split(sep))
        subgraphs[parent(path)].add_node(
            pydot.Node(name=node, label=stub(path), **attr)
        )
    for src, dst, attr in edges:
        g.add_edge(pydot.Edge(src, dst, **attr))
    return g


get_params = lambda mod: {
    p.name: getattr(mod, p.name, "?") for p in signature(type(mod)).parameters.values()
}


class DotGraph:
    colors = ColorMap()

    def __init__(self, net, size=15, direction="LR"):
        graph = build_graph(net)
        self.nodes = [
            (
                k,
                {
                    "tooltip": "%s %.1000r" % (type(n).__name__, get_params(n)),
                    "fillcolor": "#" + self.colors[type(n)],
                },
            )
            for k, (n, i) in graph.items()
        ]
        self.edges = [(src, k, {}) for (k, (n, i)) in graph.items() for src in i]
        self.size, self.direction = size, direction

    def dot_graph(self, **kwargs):
        return make_pydot(
            self.nodes, self.edges, size=self.size, direction=self.direction, **kwargs
        )

    def svg(self, **kwargs):
        return self.dot_graph(**kwargs).create(format="svg").decode("utf-8")

    try:
        import pydot

        def _repr_svg_(self):
            return self.svg()

    except ImportError:

        def __repr__(self):
            return "pydot is needed for network visualisation"


walk = lambda dict_, key: walk(dict_, dict_[key]) if key in dict_ else key


def remove_by_type(net, node_type):
    # remove identity nodes for more compact visualisations
    graph = build_graph(net)
    remap = {k: i[0] for k, (v, i) in graph.items() if isinstance(v, node_type)}
    return {
        k: (v, [walk(remap, x) for x in i])
        for k, (v, i) in graph.items()
        if not isinstance(v, node_type)
    }
