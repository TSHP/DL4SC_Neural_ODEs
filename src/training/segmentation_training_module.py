import torch
from torchmetrics import JaccardIndex

from src.training.abc_training_module import ABCTrainingModule


class SegmentationTrainingModule(ABCTrainingModule):
    def __init__(self, model, optimizer, params, num_classes) -> None:
        super().__init__(model, optimizer, params)
        self.num_classes = num_classes
        self.loss = torch.nn.CrossEntropyLoss()
        self.jaccard = JaccardIndex(task="multiclass", num_classes=self.num_classes).to(
            self.device
        )

    def compute_loss(self, inputs, masks):
        out = self.model(inputs)
        return out, self.loss(out, masks)

    def compute_mIoU(self, predictions, masks, gt_images):
        return self.jaccard(predictions, masks)

    def compute_metrics(self, predictions, masks):
        return {"mIoU": self.compute_mIoU(predictions, masks)}
