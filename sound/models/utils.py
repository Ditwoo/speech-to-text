import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OverLastDim(nn.Module):
    def __init__(self, module: nn.Module):
        super(OverLastDim, self).__init__()
        self.module = module

    def forward(self, x):
        *dims, features = x.size()
        
        reduced_dims = 1
        for shape in dims:
            reduced_dims *= shape

        x = x.reshape(reduced_dims, -1)
        x = self.module(x)
        x = x.view(*dims, -1)
        
        return x

class SequenceWise(nn.Module):
    def __init__(self, module: nn.Module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, n = x.size(0), x.size(1)
        x = x.reshape(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class InferenceBatchSoftmax(nn.Module):
    def __init__(self, batch_dim: int = -1):
        super(InferenceBatchSoftmax, self).__init__()
        self.batch_dim = batch_dim

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.log_softmax(input_, dim=self.batch_dim)
        else:
            return F.softmax(input_, dim=self.batch_dim)


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
        self.sqrt_2_div_pi = math.sqrt(2 / math.pi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(self.sqrt_2_div_pi * (x + 0.044715 * torch.pow(x, 3))))