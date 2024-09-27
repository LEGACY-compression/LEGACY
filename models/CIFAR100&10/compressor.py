import math
import torch


# K_thres= [1,0.3,0.1]  #the used threshold for topk and randk
# small_layer=10000 
# epochs_number=30
group_splits = [10000]
group_compressions = [1,1]
number_epochs = 30
gradient_norm_memory=[]
error_feedback = None
current_iter=0
accordion_check_inter = 20
accordion_K = 1 #100
batch_size=256
var_alpha =2
var_ksi = 0.999
var_r = None
var_v = None

def set_compreesion_split_and_level(group_splits_, group_compressions_, number_epochs_, method=''):
    global group_compressions, group_splits, number_epochs
    if len(group_splits_) >0:
        group_splits = [int(i) for i in group_splits_.split(',')]
    if 'mix' in method:
        group_compressions = [  [float(i)/100 for i in phase_compressions_.split(',')] for phase_compressions_ in group_compressions.split(';')]
    else:
        group_compressions = [float(i)/100 for i in group_compressions_.split(',')]
    number_epochs = number_epochs_

def init_error_feedback(all_layers):
    global error_feedback
    error_feedback = [torch.zeros_like(param).flatten() for param in all_layers if param.requires_grad ]

def init_variance_based(alpha, ksi, all_layers):
    global var_alpha, var_ksi, var_r, var_v
    var_alpha, var_ksi = alpha, ksi
    var_r=[torch.zeros_like(param).flatten() for param in all_layers if param.requires_grad ]
    var_v=[torch.zeros_like(param).flatten() for param in all_layers if param.requires_grad ]
# def set_compression_thresh_epoch(easy_compress, agg_compress, size_tresh, number_epochs):
#     global K_thres, small_layer, epochs_number
#     K_thres=[1, easy_compress/100, agg_compress/100]
#     small_layer = size_tresh
#     epochs_number = number_epochs


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



def layer_size_classification(flatten_grad):
    compression = group_compressions[-1]
    tensor_size = flatten_grad.numel()
    for ind, group_size in enumerate(group_splits):
        if tensor_size < group_size:
            compression = group_compressions[ind]
            break
    return compression


def iteration_calssification(current_epoch):
    compression = group_compressions[-1]
    epoch_portien= number_epochs/len(group_compressions)
    for ind in range(len(group_compressions)):
        if current_epoch < (ind+1)*epoch_portien:
            compression = group_compressions[ind]
            break
    return compression

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
    K=group_compressions[0]
    return topk(flatten_grad, K)

def topk_layer(flatten_grad):
    K=layer_size_classification(flatten_grad)
    return topk(flatten_grad, K)

def topk_adaptative(flatten_grad, current_epoch):
    K=iteration_calssification(current_epoch)
    return topk(flatten_grad, K)

def topk_policy(flaten_grad, current_epoch):
    K = size_iteration_classification(current_epoch, flaten_grad)
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


def accordion_init_memory(nb_layer):
    global gradient_norm_memory
    gradient_norm_memory = [0 for _ in range(nb_layer)]
def l2norm(x):
    return torch.sqrt(torch.sum(x ** 2))

def accordion_classification(flatten_grad, indx):
    global accordion_K
    current_norm = l2norm(flatten_grad)
    if current_iter%accordion_check_inter==0:
        if gradient_norm_memory[indx] > current_norm:
            accordion_K = group_compressions[0] # inside cretical regim
        else:
            accordion_K = group_compressions[1]
        gradient_norm_memory[indx] = current_norm    
    return accordion_K

def accordion_compressor(flatten_grad, indx):
    global current_iter
    K=accordion_classification(flatten_grad, indx)
    if indx ==len(gradient_norm_memory)-1:
        current_iter+=1
    return topk(flatten_grad, K)


def error_feedback_compensate(flatten_grad, indx):
    flatten_grad = flatten_grad+ error_feedback[indx]
    error_feedback[indx] = flatten_grad.clone().detach()
    return flatten_grad

def error_feedback_update(compressed_grad, indx):
    error_feedback[indx][compressed_grad > 0] = 0

def adacomp_layer(flatten_grad, indx):
    flatten_grad_G = flatten_grad+ error_feedback[indx]
    flatten_grad_H = flatten_grad_G + flatten_grad

    abs_gradient = flatten_grad_G.abs()
    g_max = abs_gradient.max()
    flatten_grad_H_abs = flatten_grad_H.abs()
    flatten_grad_G[flatten_grad_H_abs < g_max] = 0
    error_feedback[indx] = flatten_grad_G.clone().detach()
    error_feedback[indx][flatten_grad_H_abs >= g_max] = 0
    flatten_grad = flatten_grad_G
    return flatten_grad

adacomp_residue = None
def adacomp_compression(dW,  nb_bins):
    global adacomp_residue
    L = math.ceil(dW.numel() / nb_bins)
    G = dW
    H = 2 * dW 
    if adacomp_residue != None:
        G = dW + adacomp_residue
        H= H+ adacomp_residue
    for i in range(0, dW.numel(), L):
        part_G = G[i:i + L]
        part_H = H[i:i + L]
        max_abs_G = part_G.abs().max()
        part_H[part_H.abs() < max_abs_G] = 0
    processed_part = torch.where(H==0, torch.tensor(0.0, device=H.device), G)
    adacomp_residue = G -processed_part
    return processed_part




def variance_based(per_sample_grad, index):
    var_r[index]+=per_sample_grad.mean(axis=0).flatten()
    var_v[index]+=per_sample_grad.pow(2).mean(axis=0).flatten()
    mask = (var_r[index]**2) > (var_alpha * var_v[index])
    inverse_mask = ~mask
    compress_grad = var_r[index] * mask
    var_r[index] = var_r[index]* inverse_mask
    var_v[index] = var_ksi*var_v[index] * inverse_mask
    return compress_grad

# def variance_based(flatten_grad):
#     alpha = group_compressions[0]
#     tensor = flatten_grad.clone().detach()/batch_size
#     tensor_squared = tensor.pow(2)
#     r,v = 0,0
#     ind = []
#     gamma = 0.999

#     for i in range(len(tensor)):
#         r += tensor[i]
#         v += tensor_squared[i]
#         if pow(r,2) > alpha*v:
#             ind.append(i)
#             r = 0
#             v = 0
#         else:
#             v *= gamma

#     indices = torch.tensor(ind)
#     mask = torch.zeros_like(tensor)
#     mask[indices] = 1.0
#     flatten_grad= flatten_grad * mask
#     return flatten_grad