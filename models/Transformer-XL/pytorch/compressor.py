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

def set_compreesion_split_and_level(group_splits_, group_compressions_, number_epochs_):
    global group_compressions, group_splits, number_epochs
    if len(group_splits_) >0:
        group_splits = [int(i) for i in group_splits_.split(',')]
    group_compressions = [float(i)/100 for i in group_compressions_.split(',')]
    number_epochs = number_epochs_

def init_error_feedback(all_layers):
    global error_feedback
    error_feedback = [torch.zeros_like(param).flatten() for param in all_layers if param.requires_grad ]
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

def topk_compression(flatten_grad):
    K=group_compressions[0]
    return topk(flatten_grad, K)

def topk_layer(flatten_grad):
    K=layer_size_classification(flatten_grad)
    return topk(flatten_grad, K)

def topk_adaptative(flatten_grad, current_epoch):
    K=iteration_calssification(current_epoch)
    return topk(flatten_grad, K)

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

def adacomp(flatten_grad, indx):
    flatten_grad_G = flatten_grad+ error_feedback[indx]
    flatten_grad_H = flatten_grad_G + flatten_grad

    abs_gradient = flatten_grad_G.abs()
    g_max = abs_gradient.max()/group_compressions[0]
    flatten_grad_H_abs = flatten_grad_H.abs()
    flatten_grad_G[flatten_grad_H_abs < g_max] = 0
    error_feedback[indx] = flatten_grad_G.clone().detach()
    error_feedback[indx][flatten_grad_H_abs >= g_max] = 0
    flatten_grad = flatten_grad_G
    return flatten_grad