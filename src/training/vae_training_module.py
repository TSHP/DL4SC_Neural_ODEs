import torch
import torchvision

from src.training.utils.kl_divergence import KLDivLoss
from src.training.abc_training_module import ABCTrainingModule


class VAETrainingModule(ABCTrainingModule):
    def __init__(self, model, optimizer, params) -> None:
        super().__init__(model, optimizer, params)
        self.loss = torch.nn.MSELoss()
        self.kl_loss = KLDivLoss()
        self.kl_weight = params["kl_weight"]

    def compute_loss(self, inputs, labels):
        out, mu, log_var = self.model(inputs)
        return out, self.loss(out, inputs) + self.kl_weight * self.kl_loss(mu, log_var)

    def compute_metrics(self, predictions, labels):
        generated = self.model.sample(10)
        torchvision.utils.save_image(
            generated.cpu(),
            self.output_path / f"epoch_{self.epoch}_samples.png",
            nrow=10,
        )
        return {}
