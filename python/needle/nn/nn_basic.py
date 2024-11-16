"""The module.
"""
from functools import reduce
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dist = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        self.weight = Parameter(Tensor(dist))
        if bias:
            dist = init.kaiming_uniform(out_features, 1, device=device, dtype=dtype)
            # Why do you do this to me?
            self.bias = Parameter(Tensor(dist.reshape((1, out_features))))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        # x @ W + b
        mul = X @ self.weight
        return mul if self.bias is None else mul + self.bias.broadcast_to(mul.shape)


class Flatten(Module):
    def forward(self, X):
        return X.reshape((X.shape[0], reduce(lambda x, y: x * y, X.shape[1:])))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module.forward(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        summed = ops.summation(init.one_hot(logits.shape[1], y) * logits, axes=(1,))
        return ops.mean1d(ops.logsumexp(logits, axes=(1,)) - summed)

class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(Tensor(init.ones(dim, device=device, dtype=dtype, requires_grad=True)))
        self.bias = Parameter(Tensor(init.zeros(dim, device=device, dtype=dtype, requires_grad=True)))
        self.running_mean = Tensor(init.zeros(dim, device=device, dtype=dtype))
        self.running_var = Tensor(init.ones(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        def shape_shift(doctor):
            return doctor.reshape((1, x.shape[1])).broadcast_to(x.shape)
        if self.training:
            mean = ops.mean2d_wtf(x)
            x_sub_mean = x - shape_shift(mean)
            var = ops.mean2d_wtf_biased(x_sub_mean ** 2)
            one_sub_mom = 1 - self.momentum
            self.running_mean = self.running_mean * one_sub_mom + mean * self.momentum
            self.running_var = self.running_var * one_sub_mom + var * self.momentum
        else:
            x_sub_mean = x - shape_shift(self.running_mean)
            var = self.running_var
        w = shape_shift(self.weight)
        b = shape_shift(self.bias)
        return w * (x_sub_mean / shape_shift(ops.sqrt(var + self.eps))) + b

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(Tensor(init.ones(dim, device=device, dtype=dtype, requires_grad=True)))
        self.bias = Parameter(Tensor(init.zeros(dim, device=device, dtype=dtype, requires_grad=True)))

    def forward(self, x: Tensor) -> Tensor:
        def shape_shift(doctor):
            return doctor.reshape((x.shape[0], 1)).broadcast_to(x.shape)

        x_sub_mean = x - shape_shift(ops.mean2d(x))
        var = ops.mean2d(x_sub_mean ** 2)
        w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return w * (x_sub_mean / shape_shift(ops.sqrt(var + self.eps))) + b


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = Tensor(init.randb(*x.shape, p=1-self.p, device=x.device, dtype="bool"))
            return x * mask / (1 - self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)
