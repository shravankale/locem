import torch
import math

from torch.nn import functional as F

def gem(x,p=3, eps=1e-6):

    if p == math.inf or p is 'inf':

        x = F.max_pool2d(x,(x.size(-2),x.size(-1)))

    elif p == 1 and not (torch.is_tensor(p) and p.requires_grad):

        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))

    else:
        x = x.clamp(min=eps)
        x = F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    return x
