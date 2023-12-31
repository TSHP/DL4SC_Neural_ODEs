import torch
from torchmetrics import JaccardIndex

from src.viz import visualize_results
from src.training.abc_training_module import ABCTrainingModule

def dice_loss(predicted, target):
    eps = 1e-7
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice = (2.0 * intersection + eps) / (union + eps)
    loss = 1.0 - dice
    return loss


class SegmentationTrainingModule(ABCTrainingModule):
    def __init__(self, model, optimizer, params, num_classes) -> None:
        super().__init__(model, optimizer, params)
        self.num_classes = num_classes
        weights = torch.ones(21).to(self.device) * 10.
        weights[0] = 1.
        self.loss = torch.nn.CrossEntropyLoss(weight=weights)
        self.jaccard = JaccardIndex(task="multiclass", num_classes=self.num_classes).to(
            self.device
        )

    def compute_loss(self, inputs, masks):
        out = self.model(inputs)
        return out, self.loss(out, torch.argmax(masks, dim=1))

    def compute_mIoU(self, predictions, masks):
        return self.jaccard(torch.argmax(predictions, dim=1), torch.argmax(masks, dim=1)).item()

    def compute_metrics(self, predictions, masks):
        visualize_results(self.output_path / f"epoch_{self.epoch}_mask_prediction.png", predictions, masks)
        return {"mIoU": self.compute_mIoU(predictions, masks)}
