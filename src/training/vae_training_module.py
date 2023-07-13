import torch
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance

from src.training.utils.kl_divergence import KLDivLoss
from src.training.abc_training_module import ABCTrainingModule


class VAETrainingModule(ABCTrainingModule):
    def __init__(self, model, optimizer, params) -> None:
        super().__init__(model, optimizer, params)
        self.loss = torch.nn.MSELoss()
        self.kl_loss = KLDivLoss()
        self.kl_weight = params["kl_weight"]
        self.fid = FrechetInceptionDistance(feature=64, reset_real_features=True).to(self.device)

    def compute_loss(self, inputs, labels):
        out, mu, log_var = self.model(inputs)
        return out, self.loss(out, inputs) + self.kl_weight * self.kl_loss(mu, log_var)
    
    def compute_fid(self, generated, real):
        self.fid.update(real, real=True)
        self.fid.update(generated, real=False)
        return self.fid.compute()

    def compute_metrics(self, predictions, labels, gt_images):
        generated = self.model.sample(1000)
        fid_score = self.compute_fid(generated, gt_images)
        self.fid.reset()
        torchvision.utils.save_image(
            generated[:30].cpu(), self.output_path / f"epoch_{self.epoch}_samples.png", nrow=10
        )
        return {"FID": fid_score}

    def sample(self, num_samples: int = 10, epoch: int = 0):
        with torch.no_grad():
            samples = self.model.sample(num_samples)
            samples = samples.cpu()
            # Save as images
            torchvision.utils.save_image(
                samples, self.output_path / f"epoch_{epoch}_samples.png", nrow=10
            )
