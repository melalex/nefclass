from dataclasses import dataclass


@dataclass
class TrainFeedback:
    train_accuracy_history: list[float]
    train_loss_history: list[float]
    valid_accuracy_history: list[float]
    valid_loss_history: list[float]
