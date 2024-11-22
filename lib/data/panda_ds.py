from pandas import DataFrame

import torch
from torch.utils.data import Dataset


class PandasClsDs(Dataset):
    __x: torch.Tensor
    __y: torch.Tensor

    def __init__(self, x: DataFrame, y: DataFrame) -> None:
        self.__x = torch.tensor(x.values, dtype=torch.float32)
        self.__y = torch.tensor(y.values, dtype=torch.int)

    def __getitem__(self, index):
        return self.__x[index].T, self.__y[index]

    def __len__(self):
        return self.__x.shape[0]
