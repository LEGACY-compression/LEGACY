import os
import numpy as np
import torch
from torch import nn
import torchvision
from core import Cutout, FlipLR, build_graph, cat, to_numpy
from device_controller import Device_Singleton
import torchvision.transforms as transforms

torch.backends.cudnn.benchmark = True

torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = None
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = Device_Singleton.get_get_device()
def update_device(id):
    global device
    device = torch.device("cuda:"+str(id) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
def get_device():
    return device
@cat.register(torch.Tensor)
def _(*xs):
    return torch.cat(xs)

@to_numpy.register(torch.Tensor)
def _(x):
    return x.detach().cpu().numpy()  

def warmup_cudnn(model, batch_size):
    #run forward and backward pass of the model on a batch of random inputs
    #to allow benchmarking of cudnn kernels 
    batch = {
        'input': torch.Tensor(np.random.rand(batch_size,3,32,32)).to(device), 
        'target': torch.LongTensor(np.random.randint(0,10,batch_size)).to(device)
    }
    model.train(True)
    o = model(batch)
    o['loss'].sum().backward()
    model.zero_grad()
    torch.cuda.synchronize(device=device)


#####################
## dataset
#####################

def cifar10(root):
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    return {
        'train': {'data': train_set.data, 'labels': train_set.targets},
        'test': {'data': test_set.data, 'labels': test_set.targets}
    }
def cifar100(root, batch_size):
    # transform1 = transforms.Compose([transforms.RandomRotation(20), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0), ratio=(0.75, 1.33), interpolation=2), transforms.ToTensor()])
    transform1 = transforms.Compose([#128
                transforms.RandomResizedCrop(128),
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
    test_trans=transforms.Compose(
            [#136-128
                        transforms.Resize(136),
        transforms.CenterCrop(128),
                # transforms.Resize(32),
                # transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    

    CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    transform_mean, transform_std = CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(transform_mean, transform_std),
        ])
    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(transform_mean, transform_std),
        ])

    datasetplus = torchvision.datasets.CIFAR100(root=root, download=True, transform=transform_train)
    # train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=test_trans)
    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(datasetplus, batch_size=batch_size, shuffle=True,
                                                num_workers=4, pin_memory=True)

    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    return (trainloader, testloader)

def imagenet(root,batch_size):
    # root='./data/tiny-imagenet-200'
    root= '/home/mostapha.essoullami/lustre/opt_for_ml-um6p-sccs-en-tifcyqktztk/users/mostapha.essoullami/imagenet/imagenet'
    # transform_train = transforms.Compose([
    #     transforms.RandomRotation(30), transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ToTensor(),#transforms.RandomErasing(),   # Converts images to tensors
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     )
    # ])

    transform_train2 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    # transforms.RandomResizedCrop(64),  # Resize and crop
    # transforms.RandomRotation(20),      # Random rotation within [-40, 40] degrees
    # transforms.RandomHorizontalFlip(),  # Random horizontal flip
    # transforms.RandomVerticalFlip(),    # Random vertical flip
    # transforms.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
    # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=45.0, scale=(0.4, 1.6)),
    # transforms.RandomPerspective(distortion_scale=0.6, p=1, interpolation=3, fill=0),
    
    transforms.ToTensor(),   
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),           # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # trainset = torchvision.datasets.ImageFolder(root=os.path.join(root, 'train'), transform=transform_train)
    trainset2 = torchvision.datasets.ImageFolder(root=os.path.join(root, 'train'), transform=transform_train2)
    trainset= trainset2
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                num_workers=4, pin_memory=True)


    #testset = torchvision.datasets.ImageFolder(root=os.path.join(root, 'val'), transform=transform_test)
    testset = torchvision.datasets.ImageFolder(root=os.path.join(root, 'val'), transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)

    print(len(testset.classes), len(trainset.classes))
    return (trainloader, testloader)
# def imagenet(root,batch_size):
#     root='./data/tiny-imagenet-200'
#     train_transform = transforms.Compose([
#         transforms.RandomRotation(30), transforms.RandomHorizontalFlip(p=0.5),
#         # transforms.Resize(256),
#         # transforms.CenterCrop(224),
#         transforms.ToTensor(),transforms.RandomErasing(),   # Converts images to tensors
#         # transforms.Normalize(
#         #     mean=[0.485, 0.456, 0.406],
#         #     std=[0.229, 0.224, 0.225]
#         # )
#     ])
#     test_transform = transforms.Compose([
#         transforms.ToTensor(),  
#         # )
#     ])

#     train_set = torchvision.datasets.ImageFolder(root=root+'/train',  transform=train_transform)
#     test_set = torchvision.datasets.ImageFolder(root=root+'/validation',  transform=test_transform)

#     tiny_imagenet_train_loader = torch.utils.data.DataLoader(
#         train_set,
#         batch_size=batch_size,
#         shuffle=True,drop_last=True
#     )
  
#     tiny_imagenet_val_loader = torch.utils.data.DataLoader(
#         test_set,
#         batch_size=batch_size,
#         shuffle=False,
#     )
#     return (tiny_imagenet_train_loader, tiny_imagenet_val_loader)
#####################
## data loading
#####################

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )
    
    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices() 
        return ({'input': x.to(device), 'target': y.to(device).long()} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)

#####################
## torch stuff
#####################

class Identity(nn.Module):
    def forward(self, x): return x
    
class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x): 
        return x*self.weight
    
class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

class Add(nn.Module):
    def forward(self, x, y): return x + y 
    
class Concat(nn.Module):
    def forward(self, *xs): return torch.cat(xs, 1)
    
class Correct(nn.Module):
    def forward(self, classifier, target):
        return classifier.max(dim = 1)[1] == target

def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False, bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False
        
    return m



class Network(nn.Module):
    def __init__(self, net):
        self.graph = build_graph(net)
        super().__init__()
        for n, (v, _) in self.graph.items(): 
            setattr(self, n, v)

    def forward(self, inputs):
        self.cache = dict(inputs)
        for n, (_, i) in self.graph.items():
            self.cache[n] = getattr(self, n)(*[self.cache[x] for x in i])
        return self.cache

trainable_params = lambda model:filter(lambda p: p.requires_grad, model.parameters())

class TorchOptimiser():
    def __init__(self, weights, optimizer, step_number=0, **opt_params):
        self.weights = weights
        self.step_number = step_number
        self.opt_params = opt_params
        self._opt = optimizer(weights, **self.param_values())
    
    def param_values(self):
        return {k: v(self.step_number) if callable(v) else v for k,v in self.opt_params.items()}
    
    def sync_param_updates(self):
        # Synchronize the parameter values from self._opt to self.opt_params
        current_params = self._opt.param_groups[0]
        for k in self.opt_params:
            if k in current_params:
                self.opt_params[k] = current_params[k]
                
    def step(self):
        self.step_number += 1
        self._opt.param_groups[0].update(**self.param_values())
        self._opt.step()

    def __repr__(self):
        return repr(self._opt)
        
def SGD(weights, lr=0, momentum=0, weight_decay=0, dampening=0, nesterov=False):
    return TorchOptimiser(weights, torch.optim.SGD, lr=lr, momentum=momentum, 
                          weight_decay=weight_decay, dampening=dampening, 
                          nesterov=nesterov)