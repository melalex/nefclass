from typing import Optional

import numpy as np
import torch
from torch import nn
from sklearn.discriminant_analysis import StandardScaler

from lib.fuzzy.member_fun import MemberFunDef


class Antecedent(nn.Module):
    name: str
    id_to_name: list[str]

    __mem_fun: nn.ModuleList
    __scale: StandardScaler

    def __init__(self, name: str, scale: Optional[StandardScaler] = None) -> None:
        super(Antecedent, self).__init__()
        self.name = name
        self.id_to_name = []
        self.__mem_fun = nn.ModuleList()
        self.__scale = scale

    def __setitem__(self, key: str, item: MemberFunDef) -> None:
        self.id_to_name.append(key)

        if self.__scale is not None:
            col_id = np.where(self.__scale.feature_names_in_ == self.name)[0][0]
            item.scale(self.__scale.mean_[col_id], self.__scale.scale_[col_id])

        self.__mem_fun.append(item.compile())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([it(x) for it in self.__mem_fun], dim=0)

    def mem_fun_count(self) -> int:
        return len(self.__mem_fun)
