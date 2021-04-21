import torch
import copy
import math
from fgc.compressor import Compressor

def normalize(value, device="cpu"):
    new_value = []
    for t in value:
        t = t.abs()
        t -= t.min()
        t /= t.max()
        if device=="gpu":
            new_value.append(t.cuda())
        else:
            new_value.append(t.cpu())
    return new_value

class FGCCompressor(Compressor):
    def __init__(self, compress_ratio=0.5, fusing_ratio=0.8):
        super().__init__(average=True, tensors_size_are_same=False)
        self.compress_ratio = compress_ratio
        self.fusing_ratio = fusing_ratio
        self.param_groups_c = None

    def clean(self):
        self.param_groups_c = None

    def compress_by_layer(self, param):
        pass

    def compress(self, mem, gmome=None, compress=True):
        agg_gradient = copy.deepcopy(mem)
        # gradient_list = copy.deepcopy(mem)
        # agg_gradient = []
        # for i in range(len(gradient_list[0])):
        #     result = torch.stack([j[i] for j in gradient_list]).sum(dim=0)
        #     #agg_gradient.append(result / len(gradient_list))
        #     agg_gradient.append(result)

        if gmome is not None:
            n_grad = normalize(agg_gradient, device="cpu")
            n_mome = normalize(gmome, device="cpu")

        compressed_grad = []

        for t in range(len(agg_gradient)):
            tensor = agg_gradient[t].cpu()

            shape = list(tensor.size())
            tensor = tensor.flatten()
            numel = tensor.numel()

            if gmome is not None:
                tensor_a = self.fusing_ratio * n_mome[t] + (1-self.fusing_ratio) * n_grad[t]
            else:
                tensor_a = tensor

            # tensor_a = tensor.abs()
            tensor_a = tensor_a.abs()
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
                        mask = tensor.abs().cpu() >= thr.cpu()
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
                    mask = tensor.abs().cpu() >= thr.cpu()
                    selected = mask.sum()
            else:
                mask = tensor.abs().cpu() > 0
                #selected = mask.sum()

            indices, = torch.where(mask)
            values = tensor[indices]

            tensor_compressed = values.tolist()  # , indices
            ctx = shape, mask.tolist(), numel
            # tensor boolean is to big

            compressed_grad.append((tensor_compressed, ctx))
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