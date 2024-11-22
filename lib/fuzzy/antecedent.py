from typing import Any

import torch
from torch import nn


class Antecedent(nn.Module):
    name: str
    id_to_name: list[str]

    __mem_fun: nn.ModuleList

    def __init__(self, name: str) -> None:
        super(Antecedent, self).__init__()
        self.name = name
        self.id_to_name = []
        self.__mem_fun = nn.ModuleList()

    def __setitem__(self, key: str, item: nn.Module) -> None:
        self.id_to_name.append(key)
        self.__mem_fun.append(item)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([it(x) for it in self.__mem_fun], dim=0)

    def mem_fun_count(self) -> int:
        return len(self.__mem_fun)
