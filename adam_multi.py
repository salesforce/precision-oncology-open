import math
import torch
from torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    """Implements Adam algorithm.

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

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=([0., 0.9, 0.99, 0.999, 0.9999,0.99999], [0.9, 0.99, 0.999]), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
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

                state = self.state[p]
                betas = group['betas']
                beta0s = betas[0]
                beta1s = betas[1]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = {}
                    for beta0 in beta0s:
                        # Exponential moving average of gradient values
                        state['exp_avg'][beta0] = torch.zeros_like(p.data)

                    state['exp_avg_sq'] = {}
                    for beta1 in beta1s:
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'][beta1] = torch.zeros_like(p.data)

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                for beta0 in beta0s:
                    # Decay the first and second moment running average coefficient
                    state['exp_avg'][beta0].mul_(beta0).add_(1 - beta0, grad)
                for beta1 in beta1s:
                    state['exp_avg_sq'][beta1].mul_(beta1).addcmul_(1 - beta1, grad, grad)
                step_to_take = 0
                for beta0 in beta0s:
                    for beta1 in beta1s:
                        bias_correction1 = 1 - beta0 ** state['step']
                        bias_correction2 = 1 - beta1 ** state['step']
                        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                        step_to_take += step_size * state['exp_avg'][beta0] / state['exp_avg_sq'][beta1].sqrt().add_(group['eps']) / len(beta0s) / len(beta1s)

                p.data.add_(-step_to_take)
                # state['steptotake'] = -step_size * exp_avg / denom
        return loss
