import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    high = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-high, high=high, **kwargs)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)

def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", device=None, **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    high = math.sqrt(2.0) * math.sqrt(3.0 / fan_in)
    if shape is None:
        return rand(fan_in, fan_out, low=-high, high=high, device=device, **kwargs)
    else:
        return rand(*shape, low=-high, high=high, device=device, **kwargs)

def kaiming_normal(fan_in, fan_out, shape=None, nonlinearity="relu", device=None, **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    std = math.sqrt(2.0) / math.sqrt(fan_in)
    if shape is None:
        return randn(fan_in, fan_out, mean=0.0, std=std, device=device, **kwargs)
    else:
        return randn(*shape, mean=0, std=std, device=device, **kwargs)
