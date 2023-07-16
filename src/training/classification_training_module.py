import torch
from torchmetrics import Accuracy

from src.training.abc_training_module import ABCTrainingModule


class ClassificationTrainingModule(ABCTrainingModule):
    def __init__(self, model, optimizer, params, num_classes) -> None:
        super().__init__(model, optimizer, params)
        self.num_classes = num_classes
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes).to(
            self.device
        )

    def compute_loss(self, inputs, labels):
        out = self.model(inputs)
        return out, self.loss(out, labels)

    def compute_test_error(self, predictions, labels):
        return (1 - self.accuracy(predictions, labels)).item()

    def compute_metrics(self, predictions, labels):
        return {"Error": self.compute_test_error(predictions, labels)}
