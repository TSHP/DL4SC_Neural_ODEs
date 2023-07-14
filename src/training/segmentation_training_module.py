import torch
from torchmetrics import JaccardIndex

from src.viz import vizualize_segmentation_predictions
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

    def compute_mIoU(self, predictions, masks):
        return self.jaccard(torch.argmax(predictions, dim=1), torch.argmax(masks, dim=1))

    def compute_metrics(self, predictions, masks):
        batches = self.last_test_image_batch.shape[0] // 6
        for i in range(batches):
            vizualize_segmentation_predictions(self.output_path / f"epoch_{self.epoch}_mask_gt_p{i}.png", self.last_test_image_batch[6 * i:(i + 1) * 6].cpu().numpy(), torch.argmax(masks[6 * i:(i + 1) * 6], dim=1).cpu().numpy(), torch.argmax(predictions[6 * i:(i + 1) * 6], dim=1).cpu().numpy())
        return {"mIoU": self.compute_mIoU(predictions, masks)}
