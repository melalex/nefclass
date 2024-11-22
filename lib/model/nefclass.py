from dataclasses import dataclass
import torch
from torch import nn

from lib.fuzzy.antecedent import Antecedent


@dataclass(frozen=True, eq=True)
class RuleDef:
    conditions: tuple[int]

    def eval(self, fuzz_vars: list[torch.Tensor]) -> torch.Tensor:
        inference = torch.stack(
            [f[c] for c, f in zip(self.conditions, fuzz_vars)], dim=0
        )

        in_max, _ = inference.min(dim=0)

        return in_max


class InputLayer(nn.Module):
    antecedents: nn.ModuleList
    __n_feat: int

    def __init__(self, antecedents: list[Antecedent]) -> None:
        super(InputLayer, self).__init__()
        self.antecedents = nn.ModuleList(antecedents)
        self.__n_feat = len(antecedents)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.antecedents[i](x[i]) for i in range(self.__n_feat)]


class RuleLayer(nn.Module):
    rules: dict[RuleDef, int]
    __reverse_rules: list[list[RuleDef]]

    def __init__(self, num_classes: int) -> None:
        super(RuleLayer, self).__init__()
        self.rules = {}
        self.__reverse_rules = [[] for _ in range(num_classes)]

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            [self.__calculate_firing_str(rules, x) for rules in self.__reverse_rules],
            dim=1,
        )

    def __calculate_firing_str(
        self, rules: list[RuleDef], fuzz_vars: list[torch.Tensor]
    ) -> torch.Tensor:
        firing_str = torch.stack([rule.eval(fuzz_vars) for rule in rules], dim=0)

        return torch.sum(firing_str, dim=0)

    def add_rule(self, rule: RuleDef, cls: int) -> None:
        if rule not in self.rules:
            self.rules[rule] = cls
            self.__reverse_rules[cls].append(rule)

    def rule_count(self) -> int:
        return len(self.rules)


class OutputLayer(nn.Module):

    def __init__(self):
        super(OutputLayer, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class NefClass(nn.Module):
    labels: list[str]

    def __init__(self, antecedents: list[Antecedent], labels: list[str]) -> None:
        super(NefClass, self).__init__()
        self.labels = labels
        self.__input_layer = InputLayer(antecedents)
        self.__rule_layer = RuleLayer(len(labels))
        self.__output_layer = OutputLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__input_layer(x)
        x = self.__rule_layer(x)

        return self.__output_layer(x)

    def to_fuzz(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.__input_layer(x)

    def add_rule(self, rule: RuleDef, cls: int) -> None:
        self.__rule_layer.add_rule(rule, cls)

    def rule_count(self) -> int:
        return self.__rule_layer.rule_count()

    def list_rules_str(self) -> list[str]:
        result = []

        for rule, y in self.__rule_layer.rules.items():
            conditions = [
                (
                    self.__input_layer.antecedents[i].name,
                    self.__input_layer.antecedents[i].id_to_name[rule.conditions[i]],
                )
                for i in range(len(rule.conditions))
            ]

            rule_str = " and ".join(
                [str(it[0]) + " is " + str(it[1]) for it in conditions]
            )
            rule_str = "if " + rule_str + " then " + self.labels[y]

            result.append(rule_str)

        return result
