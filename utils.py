import torch
import numpy as np


class AverageCounter(object):
    def __init__(self):
        self._sum = None
        self._count = None
        self.reset()

    def reset(self):
        self._sum = 0.
        self._count = 0

    def __call__(self, value):
        if isinstance(value, torch.Tensor):
            sum_ = torch.sum(value).item()
            count = np.prod(value.size())
        elif isinstance(value, np.ndarray):
            sum_ = np.sum(value).item()
            count = np.prod(value.shape)
        elif isinstance(value, (int, float)):
            sum_ = value
            count = 1
        else:
            raise KeyError('Meter supports only numeric value')
        self._sum += sum_
        self._count += count

    @property
    def value(self):
        return self._sum / self._count if self._count > 0 else 0.


class StdevCounter(object):
    def __init__(self):
        self._sum = None
        self._square_sum = None
        self._count = None
        self.reset()

    def reset(self):
        self._sum = 0.
        self._square_sum = 0.
        self._count = 0

    def __call__(self, value):
        if isinstance(value, torch.Tensor):
            sum_ = torch.sum(value).item()
            square_sum = torch.sum(value ** 2).item()
            count = np.prod(value.size())
        elif isinstance(value, np.ndarray):
            sum_ = np.sum(value).item()
            square_sum = np.sum(value ** 2).item()
            count = np.prod(value.shape)
        elif isinstance(value, (int, float)):
            sum_ = value
            square_sum = value ** 2
            count = 1
        else:
            raise KeyError('Meter supports only numeric value')
        self._sum += sum_
        self._square_sum += square_sum
        self._count += count

    @property
    def value(self):
        if self._count == 0:
            return 1.
        square_mean = self._square_sum / self._count
        mean_square = (self._sum / self._count) ** 2
        return np.sqrt(square_mean - mean_square)
