from tqdm import tqdm

import torch
import torchvision

from src.training.abc_training_module import ABCTrainingModule


class VAETrainingModule(ABCTrainingModule):
    def __init__(self, model, optimizer, params) -> None:
        self.super().__init__(model, optimizer, params)

    def compute_loss(self, inputs, labels):
        return torch.sum([w * c(inputs, labels) for w, c in zip(self.get_loss_weights(), self.get_criteria())])
    
    def compute_metrics(self, inputs, labels):
        return {
            "Accuracy": self.compute_test_error(inputs, labels)
        }

    def sample(self, num_samples: int = 10, epoch: int = 0):
        with torch.no_grad():
            samples = self.model.sample(num_samples)
            samples = samples.cpu()
            # Save as images
            torchvision.utils.save_image(
                samples, self.output_path / f"epoch_{epoch}_samples.png", nrow=10
            )
