import datetime
import math
import os
import time
from contextlib import contextmanager
from typing import List

import numpy as np
import torch

communicated=[]

def get_communicated():
    return communicated


class Reducer:
    def __init__(self, random_seed, device, timer):
        self.rng = np.random.RandomState(random_seed)
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.device = device
        self.timer = timer

    def reduce(self, grad_in, grad_out, memory_out):
        """Return communicated bits"""
        raise NotImplementedError()

class TopKLayerSizeReducer(Reducer):
    """
    Use same amount as rank-based    train.config['group_splits'] = args.group_splits
    train.config['group_compressions'] =args.group_compressions
    """
    def __init__(self, random_seed, device, timer, group_splits='1000', group_compressions='1,0.1'):
        super().__init__(random_seed, device, timer)
        self.group_split = [int(i) for i in group_splits.split(',')]
        self.group_compressions = [float(i)/100 for i in group_compressions.split(',')]

        print("grouping by sizes", self.group_split, 'using these compressions', self.group_compressions)

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0
        communicated.append([])
        with self.timer("reduce.flatpack", verbosity=2):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            for tensor in grad_in:
                compression = self.group_compressions[-1]
                tensor_size = tensor.nelement()
                for ind, group_size in enumerate(self.group_split):
                    if tensor_size < group_size:
                        compression = self.group_compressions[ind]
                        break
                top_size = max(1, int(compression * tensor_size))
                flatgrad_size += top_size
                tensor_idx.append(tensor_idx[-1] + top_size)
                communicated[-1].append(top_size)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)

        with self.timer("reduce.topk", verbosity=2):
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
                compression = self.group_compressions[-1]
                tensor_size = tensor.nelement()
                for ind, group_size in enumerate(self.group_split):
                    if tensor_size < group_size:
                        compression = self.group_compressions[ind]
                        break
                top_size = max(1, int(compression * tensor_size))

                _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
                #_, indices = (tensor.view(-1).abs()).sort(descending = True)
                #positions = indices[:top_size]
                values = tensor.view(-1)[positions].contiguous()
                flat_values[start:end] = values
                flat_positions[start:end] = positions

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, start, end in zip(
                grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                positions = flat_positions[start:end]
                mem.data[:] = tensor
                mem.view(-1)[positions.long()] = 0.0

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]
            bits_communicated = n_bits(flat_values) + n_bits(flat_positions)
            params_transmitted = flat_values.numel()
            
        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start, end in zip(
                grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                out.data[:] = 0
                for pos, val in zip(worker_positions, worker_values):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers
        simulate_network_transfer(bits_communicated)
        return bits_communicated, params_transmitted
    

class TopKEpochReducer(Reducer):
    """
    Use same amount as rank-based
    """
    def __init__(self, random_seed, device, timer, group_compressions='1,0.1', num_epochs=300):
        super().__init__(random_seed, device, timer)
        self.group_compressions = [float(i)/100 for i in group_compressions.split(',')]
        self.num_epochs= num_epochs
        print("spliting epoch into ", len(self.group_compressions), ' and using these compressions', self.group_compressions)
        


    def reduce(self, grad_in, grad_out, memory_out, current_epoch):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0
        communicated.append([])

        compression = self.group_compressions[-1]
        epoch_portoin=self.num_epochs/len(self.group_compressions)
        for ind in range(len(self.group_compressions)):
            if current_epoch < (ind+1)*epoch_portoin:
                compression = self.group_compressions[ind]
                break
        # print("current epoch,", current_epoch, 'used compression is', 100*compression)
        with self.timer("reduce.flatpack", verbosity=2):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            for tensor in grad_in:
                top_size = max(1, int(compression * tensor.nelement()))
                flatgrad_size += top_size
                tensor_idx.append(tensor_idx[-1] + top_size)
                communicated[-1].append(top_size)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)

        with self.timer("reduce.topk", verbosity=2):
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):

                tensor_size = tensor.nelement()
                top_size = max(1, int(compression * tensor_size))
            
                _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
                #_, indices = (tensor.view(-1).abs()).sort(descending = True)
                #positions = indices[:top_size]
                values = tensor.view(-1)[positions].contiguous()
                flat_values[start:end] = values
                flat_positions[start:end] = positions

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, start, end in zip(
                grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                positions = flat_positions[start:end]
                mem.data[:] = tensor
                mem.view(-1)[positions.long()] = 0.0

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]
            bits_communicated = n_bits(flat_values) + n_bits(flat_positions)
            params_transmitted = flat_values.numel()
            
        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start, end in zip(
                grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                out.data[:] = 0
                for pos, val in zip(worker_positions, worker_values):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers
        simulate_network_transfer(bits_communicated)
        return bits_communicated, params_transmitted
    

class TopKReducer(Reducer):
    """
    Use same amount as rank-based
    """
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0
        communicated.append([])
        with self.timer("reduce.flatpack", verbosity=2):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            for tensor in grad_in:
                top_size = max(1, int(self.compression * tensor.nelement()))
                flatgrad_size += top_size
                tensor_idx.append(tensor_idx[-1] + top_size)
                communicated[-1].append(top_size)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)

        with self.timer("reduce.topk", verbosity=2):
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
                top_size = max(1, int(self.compression * tensor.nelement()))
                _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
                #_, indices = (tensor.view(-1).abs()).sort(descending = True)
                #positions = indices[:top_size]
                values = tensor.view(-1)[positions].contiguous()
                flat_values[start:end] = values
                flat_positions[start:end] = positions

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, start, end in zip(
                grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                positions = flat_positions[start:end]
                mem.data[:] = tensor
                mem.view(-1)[positions.long()] = 0.0

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]
            bits_communicated = n_bits(flat_values) + n_bits(flat_positions)
            params_transmitted = flat_values.numel()
            
        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start, end in zip(
                grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                out.data[:] = 0
                for pos, val in zip(worker_positions, worker_values):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers
        simulate_network_transfer(bits_communicated)
        return bits_communicated, params_transmitted

class GlobalTopKReducer(Reducer):
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0

        with self.timer("reduce.flatpack"):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            for tensor in grad_in:
                n = tensor.nelement()
                flatgrad_size += n
                tensor_idx.append(tensor_idx[-1] + n)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flatgrad = torch.empty(flatgrad_size, device=self.device)

            # Pack the flatgrad
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
                flatgrad[start:end] = tensor.view(-1)

        top_size = max(1, int(self.compression * flatgrad.nelement()))

        with self.timer("reduce.topk", verbosity=2):
            _, positions = torch.topk(flatgrad.abs(), top_size, sorted=False)
            values = flatgrad[positions].contiguous()

        with self.timer("reduce.set_memory", verbosity=2):
            for tensor, mem, start, end in zip(
                grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                local_positions = positions[(positions >= start) & (positions < end)] - start
                mem.data[:] = tensor
                mem.view(-1)[local_positions] = 0.0

        with self.timer("reduce.reduce", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, values, async_op=True)
                h2 = all_gather(worker_positions, positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [values]
                worker_positions = [positions]
            bits_communicated += n_bits(values) + n_bits(positions)
            params_transmitted += values.numel()

        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start, end in zip(
                grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                out.data[:] = 0.0
                for pos, val in zip(worker_positions, worker_values):
                    local_positions = pos[(pos >= start) & (pos < end)] - start
                    local_vals = val[(pos >= start) & (pos < end)]
                    out.view(-1)[local_positions] += local_vals / self.n_workers
        simulate_network_transfer(bits_communicated)
        return bits_communicated, params_transmitted

class ThreshReducer(Reducer):
     
    def __init__(self, random_seed, device, timer, thresh=0.5):
        super().__init__(random_seed, device, timer)
        self.threshold = thresh

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0
        
        tensors_compressed = []
        compressed_positions = []
        local_sizes = []

        communicated.append([])
        
        with self.timer("reduce.threshold", verbosity=2):
            for tensor in grad_in:
                positions, =  torch.where(tensor.view(-1).abs()>=self.threshold)
                values = tensor.view(-1)[positions].contiguous()
                tensors_compressed.append(values)
                compressed_positions.append(positions)
                local_sizes.append(values.numel())
                communicated[-1].append(values.numel())

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, positions in zip(
                grad_in, memory_out, compressed_positions
            ):
                mem.data[:] = tensor
                mem.view(-1)[positions] = 0.0
                
        with self.timer("reduce.flatpack", verbosity=2):
            flatgrad_size = 0
            tensor_idx = [0]
            for local_size in local_sizes:
                flatgrad_size += local_size
                tensor_idx.append(tensor_idx[-1] + local_size)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)
            
        with self.timer("reduce.flatput", verbosity=2):
            for values, positions, start, end in zip(tensors_compressed, compressed_positions, flatgrad_start_idx, flatgrad_end_idx):
                flat_values[start:end] = values
                flat_positions[start:end] = positions

        with self.timer("reduce.gather.context", verbosity=2):
            flatgrad_size = torch.tensor(flatgrad_size, device = self.device)
            flatgrad_start_idx = torch.tensor(flatgrad_start_idx, device = self.device)
            flatgrad_end_idx = torch.tensor(flatgrad_end_idx, device = self.device)
            if self.n_workers > 1:
                gathered_sizes = [torch.empty_like(flatgrad_size) for i in range(self.n_workers)]
                h1 = all_gather(gathered_sizes, flatgrad_size, async_op = True)
                gathered_start_indices = [torch.empty_like(flatgrad_start_idx) for i in range(self.n_workers)]
                h2 = all_gather(gathered_start_indices, flatgrad_start_idx, async_op = True)
                gathered_end_indices = [torch.empty_like(flatgrad_end_idx) for i in range(self.n_workers)]
                h3 = all_gather(gathered_end_indices, flatgrad_end_idx, async_op = True)
                h1.wait()
                h2.wait()
                h3.wait()
            else:
                gathered_sizes = [flatgrad_size]
                gathered_start_indices = [flatgrad_start_idx]
                gathered_end_indices = [flatgrad_end_idx]
                
        with self.timer("reduce.pad", verbosity=2):
            if self.n_workers > 1:
                max_size = max(gathered_sizes)
                if flatgrad_size != max_size:
                    padding_values = torch.empty(max_size-flatgrad_size, dtype=flat_values.dtype, device=flat_values.device)
                    padding_positions = torch.empty(max_size-flatgrad_size, dtype=flat_positions.dtype, device=flat_values.device)
                    flat_values = torch.cat((flat_values, padding_values), dim=0)
                    flat_positions = torch.cat((flat_positions, padding_positions), dim=0)
                
        with self.timer("reduce.gather.tensors", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]
            bits_communicated = n_bits(flat_values) + n_bits(flat_positions)
            params_transmitted = flat_values.numel()
        
        with self.timer("reduce.combine", verbosity=2):
            for out, start_indices, end_indices in zip(
                grad_out, zip(*gathered_start_indices), zip(*gathered_end_indices)
            ):
                out.data[:] = 0
                for pos, val, start, end in zip(worker_positions, worker_values, start_indices, end_indices):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers
        simulate_network_transfer(bits_communicated)
        return bits_communicated, params_transmitted
class AccordionTopKReducer(Reducer):
    """
    Modified from https://github.com/uw-mad-dash/Accordion
    """
    def __init__(self, random_seed, device, timer, k_low=0.1, k_high=0.99, detection_threshold=0.5, switch_freq=10):
        super().__init__(random_seed, device, timer)
        self.k_low = k_low
        self.k_high = k_high
        self.detection_threshold = detection_threshold
        self.switch_freq = switch_freq

    def reduce(self, grad_in, grad_out, memory_out, auto_scale_tensor, prev_norms, curr_norms, prev_lrs, curr_lrs, epoch_count):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        :auto_scale_tensor: tensor
        :prev_norms: list
        curr_norms: list
        prev_lrs:list
        curr_lrs:list
        """
        bits_communicated = 0
        params_transmitted = 0

        communicated.append([])
        with self.timer("reduce.autoscale", verbosity=2):
            # Determine compression ratio for the next switch_freq epochs
            if epoch_count%self.switch_freq == 0:
                print("switching")
                for i, grad in enumerate(grad_out):
                    curr_norms[i] = l2norm(grad)
                    if epoch_count == 0 or (prev_lrs[i] > curr_lrs[i]) or abs(prev_norms[i]-curr_norms[i])/prev_norms[i] > self.detection_threshold:
                        auto_scale_tensor[i] = self.k_high
                    else:
                        auto_scale_tensor[i] = self.k_low
                    prev_norms[i] = curr_norms[i]
                    prev_lrs[i] = curr_lrs[i]
                #Broadcast the low and high rank values from rank 0
                torch.distributed.broadcast(auto_scale_tensor, src=0)

        with self.timer("reduce.flatpack", verbosity=2):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            tttt=[]
            for i, tensor in enumerate(grad_in):
                top_size = max(1, int(auto_scale_tensor[i].item() * tensor.nelement()))
                flatgrad_size += top_size
                tensor_idx.append(tensor_idx[-1] + top_size)
                tttt.append(tensor.nelement())
                communicated[-1].append(top_size)
            # print(tttt)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)

        with self.timer("reduce.topk", verbosity=2):
            for i, (tensor, start, end) in enumerate(zip(grad_in, flatgrad_start_idx, flatgrad_end_idx)):
                top_size = max(1, int(auto_scale_tensor[i].item() * tensor.nelement()))
                _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
                #_, indices = (tensor.view(-1).abs()).sort(descending = True)
                #positions = indices[:top_size]
                values = tensor.view(-1)[positions].contiguous()
                flat_values[start:end] = values
                flat_positions[start:end] = positions

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, start, end in zip(
                grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                positions = flat_positions[start:end]
                mem.data[:] = tensor
                mem.view(-1)[positions.long()] = 0.0

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]
            bits_communicated = n_bits(flat_values) + n_bits(flat_positions)
            params_transmitted = flat_values.numel()
            
        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start, end in zip(
                grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                out.data[:] = 0
                for pos, val in zip(worker_positions, worker_values):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers
        simulate_network_transfer(bits_communicated)
        return bits_communicated, params_transmitted


class VarianceReducer(Reducer):
    def __init__(self, random_seed, device, timer, parameters, var_alpha=1.5, var_ksi=0.999):
        super().__init__(random_seed, device, timer)
        self.var_alpha = var_alpha
        self.var_ksi = var_ksi
        self.r = [torch.zeros_like(param).flatten() for param in parameters if param.requires_grad ]
        self.v = [torch.zeros_like(param).flatten() for param in parameters if param.requires_grad ]
        self.iter=0

    def reduce(self, parameters, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0
        communicated.append([])
        with self.timer("reduce.flatpack", verbosity=2):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            all_masks = []
            for index, tensor in enumerate(parameters):
                self.r[index]+=tensor.grad_sample.mean(axis=0).flatten()
                self.v[index]+=tensor.grad_sample.pow(2).mean(axis=0).flatten()
                mask = (self.r[index]**2) > (self.var_alpha * self.v[index])
                all_masks.append(mask)

                variance_size = int(mask.sum().item())
                flatgrad_size += variance_size
                tensor_idx.append(tensor_idx[-1] + variance_size)
                communicated[-1].append(variance_size)

            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            # print(tensor_idx)
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)
            # print("non zero size",flatgrad_size, 'layer shape', self.r[index].shape)

        with self.timer("reduce.variance", verbosity=2):
            for index, (start, end) in enumerate(zip(flatgrad_start_idx, flatgrad_end_idx)):
                if start==end:
                    continue
                mask = all_masks[index]
                positions = torch.nonzero(mask, as_tuple=False).squeeze()  # Get 1D positions
                compress_grad = self.r[index] * mask
                # Use `gather` to index based on positions
                values = compress_grad.view(-1).gather(0, positions)
                # positions = torch.nonzero(mask)

                # values = compress_grad.view(-1)[positions].contiguous()
                flat_values[start:end] = values
                flat_positions[start:end] = positions
                # Set r and v to zero where the condition is not met
                inverse_mask = ~mask
                self.r[index] = self.r[index]* inverse_mask
                self.v[index] = self.var_ksi*self.v[index] * inverse_mask

        # with self.timer("reduce.memory", verbosity=2):  # varaince compression save memory in r
        #     for tensor, mem, start, end in zip(
        #         grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
        #     ):
        #         positions = flat_positions[start:end]
        #         mem.data[:] = tensor
        #         mem.view(-1)[positions.long()] = 0.0
        with self.timer("reduce.gather.context", verbosity=2):
            flatgrad_size = torch.tensor(flatgrad_size, device = self.device)
            flatgrad_start_idx = torch.tensor(flatgrad_start_idx, device = self.device)
            flatgrad_end_idx = torch.tensor(flatgrad_end_idx, device = self.device)
            if self.n_workers > 1:
                gathered_sizes = [torch.empty_like(flatgrad_size) for i in range(self.n_workers)]
                h1 = all_gather(gathered_sizes, flatgrad_size, async_op = True)
                gathered_start_indices = [torch.empty_like(flatgrad_start_idx) for i in range(self.n_workers)]
                h2 = all_gather(gathered_start_indices, flatgrad_start_idx, async_op = True)
                gathered_end_indices = [torch.empty_like(flatgrad_end_idx) for i in range(self.n_workers)]
                h3 = all_gather(gathered_end_indices, flatgrad_end_idx, async_op = True)
                h1.wait()
                h2.wait()
                h3.wait()
            else:
                gathered_sizes = [flatgrad_size]
                gathered_start_indices = [flatgrad_start_idx]
                gathered_end_indices = [flatgrad_end_idx]
                
        with self.timer("reduce.pad", verbosity=2):
            if self.n_workers > 1:
                max_size = max(gathered_sizes)
                if flatgrad_size != max_size:
                    padding_values = torch.empty(max_size-flatgrad_size, dtype=flat_values.dtype, device=flat_values.device)
                    padding_positions = torch.empty(max_size-flatgrad_size, dtype=flat_positions.dtype, device=flat_values.device)
                    flat_values = torch.cat((flat_values, padding_values), dim=0)
                    flat_positions = torch.cat((flat_positions, padding_positions), dim=0)


        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                start_time = time.time()
                self.iter+=1
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                # all_gather(worker_values, flat_values)
                # all_gather(worker_positions, flat_positions)
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
                # torch.distributed.barrier()
                elapsed_time = time.time() - start_time
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]
           
            bits_communicated = n_bits(flat_values) + n_bits(flat_positions)
            params_transmitted = flat_values.numel()
            
        with self.timer("reduce.combine", verbosity=2):
            for out, start_indices, end_indices in zip(
                grad_out, zip(*gathered_start_indices), zip(*gathered_end_indices)
            ):
                out.data[:] = 0
                for pos, val, start, end in zip(worker_positions, worker_values, start_indices, end_indices):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers
        simulate_network_transfer(bits_communicated)
        return bits_communicated, params_transmitted



class GlobalCATReducer(Reducer):
    def __init__(self, random_seed, device, timer, c_0 = 1, c_1 = 60, P_max = 1460): #c_0 = 576, c_1 = 64, P_max = 512
        super().__init__(random_seed, device, timer)
        self.c_0 = c_0
        self.c_1 = c_1
        self.P_max = P_max

        self.tau_max = math.floor(self.P_max/64)  # the number of gradient components (+ index) that can fit within a single communication packet
        print('tau max', self.tau_max, 'P_max', P_max)
    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0
        communicated.append([])
        with self.timer("reduce.flatpack"):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            for tensor in grad_in:
                n = tensor.nelement()
                flatgrad_size += n
                tensor_idx.append(tensor_idx[-1] + n)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flatgrad = torch.empty(flatgrad_size, device=self.device)

            # Pack the flatgrad
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
                flatgrad[start:end] = tensor.view(-1)

        with self.timer("reduce.CAT", verbosity=2):
            sorted_abs_values = torch.sort(torch.abs(flatgrad), descending=True).values
            squared_values = sorted_abs_values.pow(2)
            total_sum_squares = squared_values.sum()
            log_d = math.ceil(math.log2(flatgrad.nelement()))
            max_nb_iter = math.ceil(flatgrad.nelement()/self.tau_max)
            cumulative_sum = 0.0
            previous_ratio = 0.0
            #### Packet model
            # for i in range(max_nb_iter):
            #     cumulative_sum += squared_values[i*self.tau_max: (i+1)*self.tau_max].sum()
            #     nb_element = min((i+1)*self.tau_max, flatgrad.nelement())
            #     current_ratio = cumulative_sum / (total_sum_squares*(self.communication_cost(nb_element, log_d)))
            #     if i > 0 and current_ratio < previous_ratio:
            #         break  # The current index is T since the ratio decreased
            #     previous_ratio = current_ratio

            # top_size = min((i+1)*self.tau_max, flatgrad.nelement())

            for i, value in enumerate(squared_values):
                cumulative_sum += value
                current_ratio = cumulative_sum / (total_sum_squares*(self.communication_cost(i+1, log_d)))
                if i > 0 and current_ratio < previous_ratio:
                    break  # The current index is T since the ratio decreased
                previous_ratio = current_ratio
            top_size = i
            communicated[-1].append(top_size)
            _, positions = torch.topk(flatgrad.abs(), top_size, sorted=False)
            values = flatgrad[positions].contiguous()
            # print('calculated',positions.max())
            flat_values = values
            flat_positions = positions


        with self.timer("reduce.gather.context", verbosity=2):
            flatgrad_size = torch.tensor(top_size, device = self.device)
            # flatgrad_start_idx = torch.tensor(flatgrad_start_idx, device = self.device)
            # flatgrad_end_idx = torch.tensor(flatgrad_end_idx, device = self.device)
            if self.n_workers > 1:
                gathered_sizes = [torch.empty_like(flatgrad_size) for i in range(self.n_workers)]
                h1 = all_gather(gathered_sizes, flatgrad_size, async_op = True)
                # gathered_start_indices = [torch.empty_like(flatgrad_start_idx) for i in range(self.n_workers)]
                # h2 = all_gather(gathered_start_indices, flatgrad_start_idx, async_op = True)
                # gathered_end_indices = [torch.empty_like(flatgrad_end_idx) for i in range(self.n_workers)]
                # h3 = all_gather(gathered_end_indices, flatgrad_end_idx, async_op = True)
                h1.wait()
                # h2.wait()
                # h3.wait()
            else:
                gathered_sizes = [flatgrad_size]
                # gathered_start_indices = [flatgrad_start_idx]
                # gathered_end_indices = [flatgrad_end_idx]
        with self.timer("reduce.pad", verbosity=2):
            if self.n_workers > 1:
                max_size = max(gathered_sizes)
                if flatgrad_size != max_size:
                    padding_values = torch.zeros(max_size-flatgrad_size, dtype=values.dtype, device=values.device)
                    padding_positions = torch.zeros(max_size-flatgrad_size, dtype=positions.dtype, device=positions.device)
                    flat_values = torch.cat((values, padding_values), dim=0)
                    flat_positions = torch.cat((positions, padding_positions), dim=0)
                    # print('pading', flat_positions.max)


        
        with self.timer("reduce.gather.tensors", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.zeros_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.zeros_like(flat_positions) for i in range(self.n_workers)]
                # print('before',worker_positions[0].max(), worker_positions[1].max(), flat_positions.max())
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]
            bits_communicated = n_bits(flat_values) + n_bits(flat_positions)
            params_transmitted = flat_values.numel()
        # print('after',worker_positions[0].max(), worker_positions[1].max())
        with self.timer("reduce.combine", verbosity=2):
            # Flatten the gradients into one tensor
            flat_grad = torch.cat([g.view(-1) for g in grad_out])
            # print('combine')
            flat_grad.zero_()
            for pos, val in zip(worker_positions, worker_values):
                # print(f"flat_grad size: {flat_grad.size()}, pos shape: {pos.shape}, pos max: {pos.max()}, val {val.shape}")
                flat_grad[pos] += val/ self.n_workers

            # Reshape the flattened gradient back into per-layer format
            offset = 0

            for i, g in enumerate(grad_out):
                num_elements = g.numel()
                grad_out[i] = flat_grad[offset:offset + num_elements].view_as(g)
                offset += num_elements

    
            # grad_out.data[:] = 0
            # for pos, val in zip(worker_positions, worker_values):
            #     # positions = pos[start:end]
            #     # values = val[start:end]
            #     # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
            #     grad_out.view(-1)[pos] += val / self.n_workers
            #     print(grad_out[:10], pos[:10], val[:10])
            #     print('----')
        simulate_network_transfer(bits_communicated)

        # with self.timer("reduce.memory", verbosity=2):
        #     for tensor, mem, positions in zip(
        #         grad_in, memory_out, compressed_positions
        #     ):
        #         mem.data[:] = tensor
        #         mem.view(-1)[positions] = 0.0
        # with self.timer("reduce.reduce", verbosity=2):
        #     if self.n_workers > 1:
        #         worker_values = [torch.empty_like(values) for i in range(self.n_workers)]
        #         worker_positions = [torch.empty_like(positions) for i in range(self.n_workers)]
        #         h1 = all_gather(worker_values, values, async_op=True)
        #         h2 = all_gather(worker_positions, positions, async_op=True)
        #         h1.wait()
        #         h2.wait()
        #     else:
        #         worker_values = [values]
        #         worker_positions = [positions]
        #     bits_communicated += n_bits(values) + n_bits(positions)
        #     params_transmitted += values.numel()

        # with self.timer("reduce.combine", verbosity=2):
        #     for tensor, out, start, end in zip(
        #         grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
        #     ):
        #         out.data[:] = 0.0
        #         for pos, val in zip(worker_positions, worker_values):
        #             local_positions = pos[(pos >= start) & (pos < end)] - start
        #             local_vals = val[(pos >= start) & (pos < end)]
        #             out.view(-1)[local_positions] += local_vals / self.n_workers
        # simulate_network_transfer(bits_communicated)
        return bits_communicated, params_transmitted
    
    def communication_cost(self, T, log_d):
        payload =  T*(32 + log_d)
        cost = self.c_0*(payload/self.P_max) + self.c_1   # from A Flexible Framework for Communication-Efficient Machine Learning
        return cost
        


def communication_cost(T, log_d):
    payload = T*(32 + log_d)
    cost = 576*(payload/512) + 64   # from A Flexible Framework for Communication-Efficient Machine Learning
    return cost



@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col


class ExactReducer(Reducer):
    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        with self.timer("reduce.zero_mem", verbosity=2):
            for mem in memory_out:
                mem.zero_()

        with self.timer("reduce.build_lists", verbosity=2):
            list_in = grad_in
            list_out = grad_out

        with self.timer("reduce.reduce", verbosity=2):
            bits_communicated, params_transmitted = reduce_mean_list(self.device, list_in, list_out, self.timer)
        simulate_network_transfer(bits_communicated)
        return bits_communicated, params_transmitted


def reduce_mean_list(
    device: torch.device, list_in: List[torch.Tensor], list_out: List[torch.Tensor], timer
):
    if torch.distributed.is_available():
        n_workers = torch.distributed.get_world_size()
    else:
        n_workers = 1

    if n_workers == 1:
        for t_in, t_out in zip(list_in, list_out):
            t_out[:] = t_in
        return 0,0

    with timer("reduce.mean.pack"):
        buffer = TensorBuffer(list_in)

    with timer("reduce.mean.allreduce"):
        buffer.all_reduce()
        buffer.buffer /= n_workers
        bits_communicated = buffer.bits()
        params_transmitted = buffer.nelement()

    with timer("reduce.mean.unpack", verbosity=2):
        buffer.unpack(list_out)
        
    return bits_communicated, params_transmitted


def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()

class TensorBuffer():
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """
    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors = tensors

        self.buffer = torch.cat([t.view(-1) for t in tensors]) # copies
    
    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(*self._tensors[index].shape)

    def __len__(self):
        return len(self._tensors)

    def pack(self, tensors=None):
        # Optional. init already does this.
        if tensors is None:
            tensors = self._tensors
        for tensor, entry in zip(tensors, self):
            entry[:] = tensor

    def unpack(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor[:] = entry

    def nelement(self):
        return self.buffer.nelement()

    def element_size(self):
        return self.buffer.element_size()

    def bits(self):
        return 8 * self.nelement() * self.element_size()

    def all_reduce(self, async_op=False):
        return torch.distributed.all_reduce(self.buffer, async_op=async_op)
    
    def all_gather(self, async_op=False):
        n_workers = torch.distributed.get_world_size() if torch.distributed.is_available() else 1
        buffers = [torch.empty_like(self.buffer) for i in range(n_workers)]
        handle = all_gather(buffers, self.buffer, async_op=async_op)
        if async_op:
            return buffers, handle
        else:
            return buffers
    
def simulate_network_transfer(bits_to_transfer):
    """
    Simulates the time needed to transfer a given number of bits over a 1 Gbps network.
    """
    # Network speed in bits per second
    network_speed_bps = 1e9  # 1 Gbps
    # Calculate the time needed to transfer the bits (in seconds)
    time_needed = bits_to_transfer / network_speed_bps
    # print(f'time needed to cominicate {bits_to_transfer} is {time_needed}')
    # time.sleep(time_needed)

def all_reduce(*args, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_reduce(*args, **kwargs)


def all_gather(out_list, in_tensor, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_gather(out_list, in_tensor, **kwargs)
    else:
        assert len(out_list) == 1
        out_list[0].data = in_tensor


@torch.jit.script
def l2norm(x):
    return torch.sqrt(torch.sum(x ** 2))


def normalize_(tensor):
    """Divide by L2 norm. In place"""
    tensor /= l2norm(tensor)
