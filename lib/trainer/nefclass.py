from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lib.model.nefclass import NefClass, RuleDef


class NefClassBestRuleTrainer:
    __device: torch.device

    def __init__(self, device: torch.device) -> None:
        self.__device = device

    def train(self, model: NefClass, loader: DataLoader, max_rules: int):
        all_rules = defaultdict(lambda _: defaultdict(0))

        with torch.no_grad():
            for x, y_true in tqdm(loader, desc="Create rules"):
                x = x.T.to(self.__device)
                y_true = y_true.to(self.__device)

                fuzz_vars = model.to_fuzz(x)
                terms = [torch.argmax(it, dim=0) for it in fuzz_vars]
                rules = [RuleDef(tuple(it)) for it in zip(*terms)]

                for rule, y in zip(rules, y_true):
                    all_rules[rule][y] += rule.eval(fuzz_vars)

        scores = [self.__select_max_score(r, s) for r, s in all_rules.items()]
        scores = scores.sort(reverse=True, key=lambda it: it[2])

        for rule, cls, _ in scores[:max_rules]:
            model.add_rule(rule, cls)

    def __select_max_score(
        self, rule: RuleDef, score: dict[int, float]
    ) -> tuple[RuleDef, int, float]:
        key = max(score, key=score.get)
        return rule, key, score[key]


class NefClassSimpleRuleTrainer:
    __device: torch.device

    def __init__(self, device: torch.device) -> None:
        self.__device = device

    def train(self, model: NefClass, loader: DataLoader, max_rules: int):
        with tqdm(total=max_rules, desc="Create rules") as pb, torch.no_grad():
            for x, y_true in loader:
                x = x.T.to(self.__device)
                y_true = y_true.to(self.__device)

                fuzz_vars = model.to_fuzz(x)
                terms = [torch.argmax(it, dim=0) for it in fuzz_vars]
                rules = [RuleDef(tuple(it)) for it in zip(*terms)]

                for rule, y in zip(rules, y_true):
                    model.add_rule(rule, y)

                    rule_count = model.rule_count()

                    pb.update(rule_count)

                    if rule_count >= max_rules:
                        return
