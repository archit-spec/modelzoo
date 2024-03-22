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




class SparseAdam(torch.optim.Adam):
    """Implements Adam algorithm with sparse gradients.
    Currently GPU-only. Requires Apex to be installed via
    ```
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"
    ```
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
    params (iterable): iterable of parameters to optimize or dicts defining
    parameter groups
    lr (float, optional): learning rate (default: 1e-3)
    betas (Tuple[float, float], optional): coefficients used for computing
    running averages of gradient and its square (default: (0.9, 0.999))
    eps (float, optional): term added to the denominator to improve
    numerical stability (default: 1e-8)
    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    amsgrad (boolean, optional): whether to use the AMSGrad variant of this
    algorithm from the paper `On the Convergence of Adam and Beyond`_
    (default: False)
    """

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
