import torch
from torchmetrics import Accuracy

from src.training.abc_training_module import ABCTrainingModule


class SegmentationTrainingModule(ABCTrainingModule):
    def __init__(self, model, optimizer, params) -> None:
        super().__init__(model, optimizer, params)
        self.num_classes = params["num_classes"]
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def compute_loss(self, out, labels):
        return out, self.loss(out, labels)
    
    def compute_test_error(self, inputs, labels):
        return 1 - self.accuracy(self.model(inputs), labels)
    
    def compute_metrics(self, inputs, labels):
        return {
            "Test Error": self.compute_test_error(inputs, labels)
        }
    
    



