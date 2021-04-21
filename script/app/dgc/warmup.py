import os
import numpy as np

class warmup:
    def __init__(self, start_lr=0.0001, max_lr=0.001, min_lr=0.0001, base_step=10, end_step=50):
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.base_step = base_step
        self.end_step = end_step
    
    def get_lr_from_step(self, step):
        if step < self.base_step:
            retult = self.start_lr + step*(self.max_lr - self.start_lr)/self.base_step
        elif step == self.base_step:
            retult =  self.max_lr
#         elif step > self.base_step and step < self.end_step:
#             retult =  self.max_lr - (step-self.base_step)*(self.max_lr - self.min_lr)/(self.end_step-self.base_step)
        elif step > self.base_step and step < self.end_step:
            retult =  self.max_lr - (self.max_lr - self.min_lr)*np.sin((np.pi/2.0)*(step-self.base_step)/(self.end_step-self.base_step))  
        elif step >= self.end_step:
            retult =  self.min_lr
        return round(retult, 6)