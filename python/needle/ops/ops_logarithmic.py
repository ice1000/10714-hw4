from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

def do_logsumexp(axes, Z):
    maxes = Z.max(axis=axes, keepdims=True)
    sums = array_api.exp(Z - maxes.broadcast_to(Z.shape)).sum(axis=axes, keepdims=True)
    return array_api.log(sums) + maxes

class LogSoftmax(TensorOp):
    def compute(self, Z):
        return Z - do_logsumexp((1,), Z)

    def gradient(self, out_grad, node):
        Z: Tensor = node.inputs[0]
        # I have no other idea
        maxes = Tensor(Z.max(axis=(1,), keepdims=True))
        exponent = exp(Z - maxes.broadcast_to(Z.shape))
        grad_over_e = summation(out_grad, axes=(1,)) / summation(exponent, axes=(1,))
        return (out_grad - exponent * grad_over_e.reshape((Z.shape[0], 1)).broadcast_to(Z.shape),)


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, Z):
        # log(sum(exp(Z - max(Z)))) + max(Z)
        use_axes = self.axes
        if self.axes is None:
            use_axes = (1,)
        result = do_logsumexp(use_axes, Z)
        if self.axes is None:
            new_shape = ()
        else:
            new_shape = list(Z.shape)
            for axis in reversed(sorted(self.axes)):
                new_shape.pop(axis)
        return array_api.reshape(result, new_shape)

    def gradient(self, out_grad, node):
        Z: Tensor = node.inputs[0]
        shape_prebroadcast = shape_before_broadcast(self.axes, Z.shape)
        preexp = exp(Z - reshape(node, shape_prebroadcast))
        return (reshape(out_grad, shape_prebroadcast).broadcast_to(Z.shape) * preexp,)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

