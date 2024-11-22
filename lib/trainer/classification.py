import copy
from typing import Optional

import numpy as np
import sklearn
import torch
from tqdm import tqdm
from lib.util.feedback import TrainFeedback
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


class ClassificationTrainer:
    __device: torch.device

    def __init__(self, device: torch.device) -> None:
        self.__device = device

    def train(
        self,
        model: nn.Module,
        num_epochs: int,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        loss_fun: nn.Module,
        optimizer: Optimizer,
        initial_patience: Optional[int] = None,
        gradient_clip: Optional[float] = None,
    ) -> TrainFeedback:
        num_batches = len(train_loader)
        train_accuracy_history = []
        train_loss_history = []
        valid_accuracy_history = []
        valid_loss_history = []
        best_accuracy = 0
        best_model_weights = None
        patience = initial_patience

        for epoch in range(num_epochs):
            with tqdm(total=num_batches) as p_bar:
                p_bar.set_description("Epoch [%s]" % (epoch + 1))

                train_total_loss = 0
                train_total_accuracy = 0

                for x, y_true in train_loader:
                    x = x.T.to(self.__device)
                    y_true = y_true.to(self.__device)

                    outputs = model(x)

                    y_pred = self.__extract_prediction(outputs.data)

                    accuracy = self.__calculate_accuracy(y_true, y_pred)

                    train_total_accuracy += accuracy

                    loss = loss_fun(outputs, y_true)

                    train_total_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()

                    if gradient_clip != None:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

                    optimizer.step()

                    p_bar.update()
                    p_bar.set_postfix(loss=loss.item(), accuracy=accuracy)

                train_avg_loss = train_total_loss / num_batches
                train_avg_accuracy = train_total_accuracy / num_batches

                train_loss_history.append(train_avg_loss)
                train_accuracy_history.append(train_avg_accuracy)

                valid_loss, valid_accuracy, valid_f1, _ = self.eval(
                    model, loss_fun, valid_loader
                )

                valid_accuracy_history.append(valid_accuracy)
                valid_loss_history.append(valid_loss)

                progress_postfix = {
                    "loss": train_avg_loss,
                    "accuracy": train_avg_accuracy,
                    "valid_loss": valid_loss,
                    "valid_accuracy": valid_accuracy,
                    "valid_f1": valid_f1,
                }

                # Early stopping
                if patience != None:
                    if valid_accuracy > best_accuracy:
                        best_accuracy = valid_accuracy
                        best_model_weights = copy.deepcopy(model.state_dict())
                        patience = initial_patience
                    else:
                        patience -= 1
                    progress_postfix["patience"] = patience

                p_bar.set_postfix(**progress_postfix)

                if patience == 0:
                    break

        if best_model_weights is not None:
            model.load_state_dict(best_model_weights)

        return TrainFeedback(
            train_accuracy_history,
            train_loss_history,
            valid_accuracy_history,
            valid_loss_history,
        )

    def eval(
        self,
        model: nn.Module,
        loss_fun: nn.Module,
        loader: DataLoader,
        record_class_stats: bool = False,
    ) -> tuple[float, float, np.array]:
        with torch.no_grad():
            total_loss = 0
            total_accuracy = 0
            total_f1 = 0
            batch_count = len(loader)
            confusion_matrix = None

            for x, y in loader:
                x = x.T.to(self.__device)
                y = y.to(self.__device)
                outputs = model(x)
                total_loss += loss_fun(outputs, y).item()
                y_predicted = self.__extract_prediction(outputs.data)
                total_accuracy += self.__calculate_accuracy(y, y_predicted)
                total_f1 += f1_score(y.cpu(), y_predicted.cpu())

                if record_class_stats:
                    if confusion_matrix is None:
                        num_classes = outputs.data.shape[1]
                        confusion_matrix = np.zeros(
                            (num_classes, num_classes), dtype=np.uint8
                        )

                    for t, p in zip(y.view(-1), y_predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

            return (
                total_loss / batch_count,
                total_accuracy / batch_count,
                total_f1 / batch_count,
                confusion_matrix,
            )

    def __calculate_accuracy(self, y_true: torch.Tensor, y_predicted: torch.Tensor):
        return (y_predicted == y_true).sum().item() / y_true.size(0)

    def __extract_prediction(self, y_predicted: torch.Tensor):
        _, predicted = torch.max(y_predicted, dim=1)
        return predicted
