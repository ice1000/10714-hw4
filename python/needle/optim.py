"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        p: ndl.nn.Parameter
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.detach()
            if self.weight_decay != 0:
                grad += self.weight_decay * p.data

            if p not in self.u:
                self.u[p] = ndl.zeros_like(grad)

            self.u[p] = self.u[p] * self.momentum + grad * (1 - self.momentum)
            p.data -= self.lr * self.u[p]

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        p: ndl.nn.Parameter
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.detach()
            if self.weight_decay != 0:
                grad += self.weight_decay * p.data

            if p not in self.m:
                self.m[p] = ndl.zeros_like(grad)
                self.v[p] = ndl.zeros_like(grad)

            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * grad
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * grad ** 2

            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (ndl.ops.sqrt(v_hat) + self.eps)
