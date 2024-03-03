import torch.nn as nn
import torch.optim


class MomentumOptimizer(torch.optim.Optimizer):       
    
    def __init__(self, params, lr=1e-3, momentum=0.6): 
        super(MomentumOptimizer, self).__init__(params, defaults={'lr': lr}) 
        self.momentum = momentum 
        self.state = dict() 
        for group in self.param_groups: 
            for p in group['params']: 
                self.state[p] = dict(mom=torch.zeros_like(p.data)) 
      
    def step(self): 
        for group in self.param_groups: 
            for p in group['params']: 
                if p not in self.state: 
                    self.state[p] = dict(mom=torch.zeros_like(p.data)) 
                mom = self.state[p]['mom'] 
                mom = self.momentum * mom - group['lr'] * p.grad.data  
                p.data += mom