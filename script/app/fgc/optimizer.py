import torch
import copy, os, time
from fgc.fgc import FGCCompressor

"""
Original usage:

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
optimizer.zero_grad()
output = model(input)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()
"""

"""
FGCSGD usage:

optimizer = FGCSGD(model.parameters(), lr=0.1, compress_ratio=0.5)

optimizer.memory.clean()

for input,target in dataloader:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    # gradient accumulation
    optimizer.gradient_collect()

optimizer.compress()
cg = optimizer.get_compressed_gradient()
<send gradient>

if <receive aggregated gradient>:
    dg = optimizer.decompress(new_gradient)
    optimizer.set_gradient(dg)
    optimizer.step()
"""


# copy from torch/optim/sgd.py
class FGCSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=None, dgc_momentum=0.9, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, compress_ratio=0.5, fusing_ratio=0.5, checkpoint=False, device=torch.device("cpu")):
        if lr is None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.memory = FGCMemory(momentum=dgc_momentum, device=device)
        self.compressor = FGCCompressor(compress_ratio=compress_ratio, fusing_ratio=fusing_ratio, device=device)
        self.checkpoint = checkpoint
        self.device = device

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FGCSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FGCSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def memory_checkpoint_save(self):
        m_ = [i.cpu() for i in self.compressor.compress(mem=self.memory.momentums, compress=False)]
        v_ = [i.cpu() for i in self.compressor.compress(mem=self.memory.velocities, compress=False)]
        checkpoint = {"momentums": m_,
                      "velocities": v_}
        torch.save(self.memory, "/tmp/memory_checkpoint")

    def memory_checkpoint_restore(self):
        if not os.path.exists("/tmp/memory_checkpoint"):
            return
        try:
            checkpoint = torch.load("/tmp/memory_checkpoint")
            m_ = [i.to(self.device) for i in self.compressor.decompress(checkpoint['momentums'])]
            v_ = [i.to(self.device) for i in self.compressor.decompress(checkpoint['velocities'])]
            self.memory.momentums = m_
            self.memory.velocities = v_
        except:
            self.memory.momentums = None
            self.memory.velocities = None

    def gradient_collect(self):
        self.memory.add(self.param_groups)

    def get_gradient(self):
        g = []
        for group in self.param_groups:
            for p in group['params']:
                # g.append(copy.deepcopy(p.grad).cpu())
                g.append(copy.deepcopy(p.grad))
        return g

    def compress(self, global_momentum=None, compress=True, momentum_correction=False):
        # r = self.compressor.compress(self.memory.get_mem(), mome=mome, compress=compress, fusing=fusing)
        # self.memory.set_compressed_mem(r)

        if momentum_correction:
            if self.checkpoint:
                self.memory_checkpoint_restore()
            gm = self.memory.add_mem(avg=False)
            print("compensate, {}".format(time.time()))
            m = self.memory.compensate(gm)
            print("compress, {}".format(time.time()))
            r = self.compressor.compress(m, gmome=global_momentum, compress=compress)
            print("compress done, {}".format(time.time()))
            self.memory.update(r)

            if self.checkpoint:
                self.memory_checkpoint_save()
        else:
            r = self.compressor.compress(self.memory.add_mem(avg=False), gmome=global_momentum, compress=compress)
        self.memory.set_compressed_mem(r)

    def decompress(self, d):
        d = self.compressor.decompress(d)
        self.memory.set_decompressed_mem(d)
        return d

    def get_compressed_gradient(self):
        return self.memory.compressed_mem

    def set_gradient(self, cg):
        agged_grad = copy.deepcopy(cg)
        for group in self.param_groups:
            for p in range(len(group['params'])):
                #print("group: {}".format(type(group['params'][p].grad)))
                #print("agged: {}".format(type(agged_grad[p])))
                if group['params'][p].is_cuda:
                    group['params'][p].grad = copy.deepcopy(agged_grad[p]).cuda()
                else:
                    group['params'][p].grad = copy.deepcopy(agged_grad[p]).cpu()

        # for group in len(self.param_groups):
        #     for p in len(self.param_groups[group]['params']):
        #         if self.param_groups[group]['params'][p].grad.size() == data[group]['params'][p].grad.size():
        #             self.param_groups[group]['params'][p].grad = copy.deepcopy(data[group]['params'][p].grad)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        #         self.gradient_collect()
        #         self.zero_grad()
        #         self.compress(compress=False)
        #         cg = self.decompress(self.get_compressed_gradient())
        #         #optimizer.set_gradient(cg)
        #         #m = self.memory.get_mem()[0]
        #         self.set_gradient(cg)

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])

        # self.memory.clean()
        return loss


class FGCMemory:
    def __init__(self, momentum=0.9, device=torch.device("cpu")):
        self.mem = []
        self.avg_mem = []
        self.compressed_mem = None
        self.decompressed_mem = None
        self.can_add = True
        self.momentum = momentum

        self.momentums = None
        self.velocities = None
        self.device = device

    def add_mem(self, mem=None, avg=False):
        if mem is None:
            gradient_list = copy.deepcopy(self.mem)
        else:
            gradient_list = copy.deepcopy(mem)
        if len(gradient_list) == 1:
            return gradient_list[0]

        avg_gradient = []
        for i in range(len(gradient_list[0])):
            result = torch.stack([j[i].to(self.device) for j in gradient_list]).sum(dim=0)
            if avg:
                avg_gradient.append(result / len(gradient_list))
            else:
                avg_gradient.append(result)
        return avg_gradient

    def compensate(self, gradient):
        avg_gradient = [i.to(self.device) for i in gradient]

        if self.momentums is None and self.velocities is None:
            self.momentums = avg_gradient
            self.velocities = self.momentums
            vec = self.velocities
        else:
            mmt = self.momentums
            vec = self.velocities
            m_e = []
            v_e = []
            for m, v, g in zip(mmt, vec, avg_gradient):
                m_ = copy.deepcopy(m).to(self.device)
                v_ = copy.deepcopy(v).to(self.device)
                m_.mul_(self.momentum).add_(g.to(self.device))
                v_.add_(m_)

                m_e.append(m_)
                v_e.append(v_)

            self.momentums = m_e
            self.velocities = v_e

        return self.velocities

    def update(self, com_gradient):
        m_n = copy.deepcopy(self.momentums)
        m_n = [i.to(self.device) for i in m_n]
        v_n = copy.deepcopy(self.velocities)
        v_n = [i.to(self.device) for i in v_n]

        m_e = []
        v_e = []

        for j, m, v in zip(com_gradient, m_n, v_n):
            new_mem, ctx = j
            shape, mask, numel = ctx
            indices, = torch.where(torch.BoolTensor(mask).to(self.device))
            m_ = m.view(-1).index_fill_(0, indices, 0)
            v_ = v.view(-1).index_fill_(0, indices, 0)
            m_e.append(copy.deepcopy(m_.view(shape)))
            v_e.append(copy.deepcopy(v_.view(shape)))

        self.momentums = m_e
        self.velocities = v_e

    def set_compressed_mem(self, d):
        # self.can_add = False
        self.compressed_mem = d
        pass

    def set_decompressed_mem(self, d):
        self.decompressed_mem = d
        pass

    def add(self, d):
        if self.can_add:
            g = []
            for group in d:
                for p in group['params']:
                    # g.append(copy.deepcopy(p.grad).cpu())
                    g.append(copy.deepcopy(p.grad))
            self.mem.append(g)
        self.mem = [self.add_mem(mem=self.mem, avg=False)]

    def get_mem(self):
        self.can_add = False
        return self.mem

    def get_compressed_mem(self):
        return self.compressed_mem

    def clean(self):
        self.mem = []
        self.compressed_mem = None
        self.decompressed_mem = None
        self.can_add = True
