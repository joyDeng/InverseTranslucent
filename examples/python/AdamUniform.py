# from numpy.core.arrayprint import dtype_is_implied
import torch
from torch.optim.optimizer import Optimizer, required
import math
from torch import Tensor
from typing import List, Optional

# from largesteps.parameterize import to_differential, from_differential
from largesteps.solvers import CholeskySolver, solve
# import scipy
# from scipy.sparse import coo_matrix, eye
# from scipy.sparse.linalg import factorized



def uniformAdam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         quantity_us: List[Tensor],
         solvers: List[callable],
         state_steps: List[int],
         largestep: bool,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):
        if largestep:
            grad = solve(solvers[i], grads[i])
        else:
            grad = grads[i]

        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        # print(step)
        quantity_u = quantity_us[i]
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        
        # denom = (exp_avg_sq.sqrt().max() / math.sqrt(bias_correction2)).add_(eps)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        if largestep:
            quantity_u.addcdiv_(exp_avg, denom, value=-step_size)
            x = solve(solvers[i], quantity_u)
            param.mul_(0.0).add_(x)
            del x
        else:
            param.addcdiv_(exp_avg, denom, value=-step_size)
            # print(param)
            # print(grad)
            # print(exp_avg)
            # print(denom)
            # print(step_size)
        
        
        

class UAdam(Optimizer):
    """Implement the algorithm propsed in Large steps in Inverse Rendering of Geometry """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), lambdas=0.1, eps=1e-8, weight_decay=0, amsgrad=False, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        # if not len(params) == len(jacobians) and len(params) == len(jacobians_i):
            # raise ValueError(" Invalid jacobian size len(params) = {} vs  len(jacobian) {} ".format(len(params), len(jacobians)))
        
        defaults = dict(lr=lr, betas=betas, lambdas=lambdas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize, I_Ls=None, largestep=False)
        super(UAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(UAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def setJacobian(self, IL):
        for i, group in enumerate(self.param_groups):
            if group['largestep'] == True:
                group['I_Ls'] = IL[i]

    def setLearningRate(self, lrs):
        for i, group in enumerate(self.param_groups):
            group['lr'] = lrs[i]
            

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:            
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            quantity_u = []
            solvers = []
            state_steps = []
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]

                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        if group['largestep']:
                            state['solver'] = CholeskySolver(group['I_Ls'])
                            state['quantity_u'] = group['I_Ls'] @ p #torch.tensor(u, device=p.get_device(),dtype=p.dtype)
                            del group['I_Ls']
                            group['I_Ls'] = 0
                        else:
                            state['solver'] = None
                            state['quantity_u'] = None

                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    solvers.append(state['solver'])
                    quantity_u.append(state['quantity_u'])
        
                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state['step'] += 1

                    state_steps.append(state['step'])

            uniformAdam(params_with_grad, 
                        grads, 
                        exp_avgs, 
                        exp_avg_sqs, 
                        max_exp_avg_sqs,
                        quantity_u,
                        solvers,
                        state_steps, 
                        largestep=group['largestep'],
                        amsgrad=group['amsgrad'],
                        beta1=beta1,
                        beta2=beta2,
                        lr=group['lr'],
                        weight_decay=group['weight_decay'],
                        eps=group['eps'])

        return loss



