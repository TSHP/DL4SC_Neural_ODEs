from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision

from training.abc_training_module import ABCTrainingModule
from src.utils import kl_div
from src.constants import INPUT_DIR, MODEL_DIR


class VAETrainingModule(ABCTrainingModule):
    def __init__(self, model, optimizer, params) -> None:
        self.super().__init__(model, optimizer, params)
        self.kl_strength = params["kl_strength"]

    def fit(self, num_epochs: int = 100):
        for cur_epoch in (pbar_epoch := tqdm(range(num_epochs))):
            running_loss = 0.0
            running_loss_kl = 0.0
            for _, (images, _) in enumerate(self.training_dataloader):
                images = images.to(self.device)
                out, mu, logvar = self.model(images)
                kl_loss = kl_div(mu, logvar)
                loss = self.criterion(out, images) + self.kl_strength * kl_loss
                running_loss += loss.item()
                running_loss_kl += kl_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.sample(num_samples=10, epoch=cur_epoch)
            pbar_epoch.set_description(
                f"Epoch[{cur_epoch + 1}/{num_epochs}], Running Loss: {running_loss / len(self.training_dataloader):.4f}, Running KL-Loss: {running_loss_kl / len(self.training_dataloader):.4f}"
            )

    def eval(self):
        running_loss = 0.0
        with torch.no_grad():
            for _, (images, _) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                out, mu, logvar = self.model(images)
                kl_loss = kl_div(mu, logvar)
                loss = self.criterion(out, images) + self.kl_strength * kl_loss
                running_loss += loss.item()

        print(
            f"Loss of the network on test MNIST: {running_loss / len(self.test_dataloader)}%"
        )
        print(f"Number of parameters: {self.model.get_num_params()}")

    def save_model(self, tag: str = "last"):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.output_path / f"{tag}_model.pt")

    def sample(self, num_samples: int = 10, epoch: int = 0):
        with torch.no_grad():
            samples = self.model.sample(num_samples)
            samples = samples.cpu()
            # Save as images
            torchvision.utils.save_image(
                samples, self.output_path / f"epoch_{epoch}_samples.png", nrow=10
            )
