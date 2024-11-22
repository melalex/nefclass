from typing import Callable
import numpy as np
import torch

from torch import nn, Tensor


class GaussMf(nn.Module):
    __mean: nn.Parameter
    __std: nn.Parameter

    def __init__(self, mean: float, std: float) -> None:
        super(GaussMf, self).__init__()
        self.__mean = nn.Parameter(torch.tensor(mean, dtype=torch.float32))
        self.__std = nn.Parameter(torch.tensor(std, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-((x - self.__mean) ** 2.0) / (2 * self.__std**2.0))


class SingletonMf(nn.Module):
    __value: float

    def __init__(self, value: float) -> None:
        super(SingletonMf, self).__init__()
        self.__value = value

    def forward(self, x: Tensor) -> Tensor:
        return x == self.__value


class SShapedMf(nn.Module):
    __a: nn.Parameter
    __b: nn.Parameter

    def __init__(self, a: float, b: float) -> None:
        super(SShapedMf, self).__init__()
        self.__a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.__b = nn.Parameter(torch.tensor(b, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        a = self.__a
        b = self.__b

        y = torch.ones(len(x)).to(x.device)
        idx = x <= a
        y[idx] = 0

        idx = torch.logical_and(a <= x, x <= (a + b) / 2.0)
        y[idx] = 2.0 * ((x[idx] - a) / (b - a)) ** 2.0

        idx = torch.logical_and((a + b) / 2.0 <= x, x <= b)
        y[idx] = 1 - 2.0 * ((x[idx] - b) / (b - a)) ** 2.0

        return y


class ZShapedMf(nn.Module):
    __a: nn.Parameter
    __b: nn.Parameter

    def __init__(self, a: float, b: float) -> None:
        super(ZShapedMf, self).__init__()
        self.__a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.__b = nn.Parameter(torch.tensor(b, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        a = self.__a
        b = self.__b

        y = torch.ones(len(x)).to(x.device)

        idx = torch.logical_and(a <= x, x < (a + b) / 2.0)
        y[idx] = 1 - 2.0 * ((x[idx] - a) / (b - a)) ** 2.0

        idx = torch.logical_and((a + b) / 2.0 <= x, x <= b)
        y[idx] = 2.0 * ((x[idx] - b) / (b - a)) ** 2.0

        idx = x >= b
        y[idx] = 0

        return y
