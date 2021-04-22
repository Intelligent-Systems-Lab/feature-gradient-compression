import torch
import copy
import math
from dgc.compressor import Compressor


class DGCCompressor(Compressor):
    def __init__(self, compress_ratio=0.5, memory=None):
        super().__init__(average=True, tensors_size_are_same=False)
        self.compress_ratio = compress_ratio
        self.param_groups_c = None
        self.memory = memory
        
    def clean(self):
        self.param_groups_c = None

    def compress_by_layer(self, param):
        pass

    def compress(self, mem=None, compress=True, momentum_correction = False):
        if mem is None:
            mem = self.memory.add_mem(self.memory.mem)

        if momentum_correction:
            agg_gradient = self.memory.compensate(mem)
        else:
            agg_gradient = mem

        compressed_grad = []

        for tensor in agg_gradient:
            tensor = tensor.cpu()
            shape = list(tensor.size())
            tensor = tensor.flatten()
            numel = tensor.numel()

            tensor_a = tensor.abs()
            tensor_a = tensor_a[tensor_a > 0]

            if not len(tensor_a)==0:
                tmin = min(tensor_a)
                tmax = max(tensor_a)
            else:
                compress = False

            if compress:
                if not len(tensor_a) == 0:
                    for i in range(10):
                        thr = (tmax + tmin) / 2
                        mask = tensor.abs() >= thr
                        selected = mask.sum()

                        if selected > (tensor_a.numel() * min(self.compress_ratio + 0.05, 1)):
                            tmin = thr
                            continue
                        if selected < (tensor_a.numel() * max(self.compress_ratio - 0.05, 0.01)):
                            tmax = thr
                            continue
                        break
                else:
                    thr = 1  # becauce all element are 0, set thr=1 make mask mask out everything
                    mask = tensor.abs() >= thr
                    selected = mask.sum()
            else:
                mask = tensor.abs() > 0
                #selected = mask.sum()

            indices, = torch.where(mask)
            values = tensor[indices]

            tensor_compressed = values.tolist()  # , indices
            ctx = shape, mask.tolist(), numel
            # tensor boolean is to big

            compressed_grad.append((tensor_compressed, ctx))
        
        if momentum_correction:
            self.memory.update(compressed_grad)
        return compressed_grad

    def decompress(self, mem):
        agg_gradient = copy.deepcopy(mem)
        decompressed_mem = []
        for j in agg_gradient:
            new_mem, ctx = j
            shape, mask, numel = ctx


            values = torch.tensor(new_mem)
            indices = torch.tensor([i for i in range(len(mask)) if mask[i]]).type(torch.long)
            mask = torch.tensor(mask)

            tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
            tensor_decompressed.scatter_(0, indices, values)
            decompressed_mem.append(tensor_decompressed.view(shape))
        return decompressed_mem

