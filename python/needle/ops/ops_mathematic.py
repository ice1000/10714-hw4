"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy
from functools import reduce

from.. init import ones

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        pass
        
    def gradient(self, out_grad, node):
        pass

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a.__pow__(self.scalar)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return (out_grad * self.scalar * a.__pow__(self.scalar - 1),)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a.__truediv__(b)

    def gradient(self, out_grad, node: Tensor):
        a, b = node.inputs
        return divide(out_grad, b), -out_grad * a / (b * b)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a.__truediv__(self.scalar)

    def gradient(self, out_grad, node: Tensor):
        return (divide_scalar(out_grad, self.scalar),)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if axes is None:
            # Swap the last two axes if no axes are specified
            self.axes = (-1, -2)

    def compute(self, a):
        new_axes = list(range(len(a.shape)))
        tmp = new_axes[self.axes[0]]
        new_axes[self.axes[0]] = new_axes[self.axes[1]]
        new_axes[self.axes[1]] = tmp
        return a.permute(new_axes)

    def gradient(self, out_grad, node):
        return (transpose(out_grad, axes=self.axes),)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        return (reshape(out_grad, node.inputs[0].shape),)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.broadcast_to(self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape

        axes = []
        for i in range(len(self.shape)):
            if i >= len(input_shape) or input_shape[i] == 1:
                axes.append(i)

        return (summation(out_grad, axes=tuple(axes)),)

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

def shape_before_broadcast(axes, input_shape):
    if axes is None:
        shape_prebroadcast = [1] * len(input_shape)
    else:
        shape_prebroadcast = list(input_shape)
        if isinstance(axes, int):
            axes = (axes,)
        for axis in axes:
            shape_prebroadcast[axis] = 1
    return shape_prebroadcast

class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        shape_prebroadcast = shape_before_broadcast(self.axes, input_shape)
        return (reshape(out_grad, shape_prebroadcast).broadcast_to(input_shape),)

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        a_expanded = None
        b_expanded = None
        if len(a.shape) > len(b.shape):
            a_expanded = a.shape[:-len(b.shape)]
            b = broadcast_to(b, a_expanded + b.shape)
        if len(b.shape) > len(a.shape):
            b_expanded = b.shape[:-len(a.shape)]
            a = broadcast_to(a, b_expanded + a.shape)

        grad_a = out_grad @ transpose(b)
        grad_b = transpose(a) @ out_grad
        # If b was expanded, reduce the extra dimensions after multiplication
        if b_expanded is not None:
            grad_a = summation(grad_a, axes=tuple(range(len(b_expanded))))
        if a_expanded is not None:
            grad_b = summation(grad_b, axes=tuple(range(len(a_expanded))))

        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)

def negate(a):
    return MulScalar(-1)(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return (out_grad / node.inputs[0],)


def log(a) -> Tensor:
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return (out_grad * node,)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        input_a = node.inputs[0].realize_cached_data()
        return (out_grad * (input_a > 0),)

def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        return (out_grad * (-(node * node) + 1),)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        # def at(i):
        #     arr = [0] * len(args)
        #     arr[i] = 1
        #     return NDArray(arr, device=args[0].device)
        # return reduce(lambda a, b: a + b, [x.reshape(x.shape + (1,)) @ at(i) for i, x in enumerate(args)])
        # the above implementation requires matmul to work for batched tensors
        return NDArray(numpy.stack([x.numpy() for x in args], axis=self.axis))

    def gradient(self, out_grad, node):
        return (split(out_grad, axis=self.axis),)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        def removeAt(shape):
            return shape[:self.axis] + shape[self.axis + 1:]
        return tuple(NDArray(x, device=A.device).reshape(removeAt(x.shape)) for x in numpy.split(A.numpy(), A.shape[self.axis], axis=self.axis))

    def gradient(self, out_grad, node):
        return stack(out_grad, axis=self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.flip(self.axes)

    def gradient(self, out_grad, node):
        return (flip(out_grad, self.axes),)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        if self.dilation == 0:
            return a
        new_shape = list(a.shape)
        for axis in self.axes:
            if axis < a.ndim:
               new_shape[axis] = new_shape[axis] * (self.dilation + 1)
        result = array_api.full(new_shape, 0, device=a.device)
        slices = [slice(None)] * a.ndim
        for axis in self.axes:
            if axis < a.ndim:
                slices[axis] = slice(None, None, self.dilation + 1)
        result[tuple(slices)] = a
        return result

    def gradient(self, out_grad, node):
        return (undilate(out_grad, self.axes, self.dilation),)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        if self.dilation == 0:
            return a
        slices = [slice(None)] * a.ndim
        for axis in self.axes:
            if axis < a.ndim:
                slices[axis] = slice(None, None, self.dilation + 1)
        return a[tuple(slices)]

    def gradient(self, out_grad, node):
        return (dilate(out_grad, self.axes, self.dilation),)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)

def mean1d(a: Tensor) -> Tensor:
    return summation(a, axes=None) / a.shape[0]

def mean2d(a: Tensor) -> Tensor:
    return summation(a, axes=(1,)) / a.shape[1]

def mean2d_wtf(a: Tensor) -> Tensor:
    return summation(a, axes=(0,)) / a.shape[0]

def mean2d_wtf_biased(a: Tensor) -> Tensor:
    return summation(a, axes=(0,)) / a.shape[0]

def sqrt(a: Tensor) -> Tensor:
    return power_scalar(a, 0.5)
