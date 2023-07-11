from pathlib import Path
from tqdm.autonotebook import tqdm

import torch
import torch.nn as nn
import torchvision
from src.constants import INPUT_DIR, MODEL_DIR


class AETrainingModule:
    def __init__(self, model, optimizer, params) -> None:
        self.model = model
        self.optimizer = optimizer

        self.output_path = Path(params["output_path"])
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.batch_size = params["batch_size"]
        self.kl_strength = params["kl_strength"]

        #self.mnist = torchvision.datasets.MNIST(root=INPUT_DIR, train=True,
        #                            download=True, transform=torchvision.transforms.ToTensor())
        self.training_dataset = torchvision.datasets.MNIST(root=INPUT_DIR, train=True,
                                    download=True, transform=torchvision.transforms.ToTensor())
        #self.training_dataset, self.validation_dataset = torch.utils.data.random_split(self.mnist, [50000, 10000])
        self.test_dataset = torchvision.datasets.MNIST(root=INPUT_DIR, train=False,
                                    download=True, transform=torchvision.transforms.ToTensor())
        
        self.training_dataloader = torch.utils.data.DataLoader(self.training_dataset, batch_size=self.batch_size,
                                      shuffle=True)
        #self.validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size,
        #                              shuffle=False)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                      shuffle=False)

        self.criterion = nn.MSELoss()

        print(f"Number of parameters: {self.model.get_num_params()}")

        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        
        print("Using device:", self.device)
        self.model.to(self.device)

    def fit(self, num_epochs: int = 100):
        total_steps = len(self.training_dataloader)
        for cur_epoch in (pbar_epoch := tqdm(range(num_epochs))):
            running_loss = 0.0
            for i, (images, _) in (pbar := tqdm(enumerate(self.training_dataloader))):
                images = images.to(self.device)
                out, mu, logvar = self.model(images)
                kl_loss = self.model.kl_loss(mu, logvar)
                loss = self.criterion(out, images) + self.kl_strength * kl_loss
                running_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_description(f'Epoch[{cur_epoch + 1}/{num_epochs}], Step[{i + 1}/{total_steps}], Loss: {loss.item():.4f}, KL-Loss: {kl_loss.item():.4f}')

            self.sample(num_samples=10, epoch=cur_epoch)
            pbar_epoch.set_description(f'Epoch[{cur_epoch + 1}/{num_epochs}], Running Loss: {running_loss / len(self.training_dataloader):.4f}')      

    def eval(self):
        running_loss = 0.0
        with torch.no_grad():
            for _, (images, _) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                out, mu, logvar = self.model(images)
                kl_loss = self.model.kl_loss(mu, logvar)
                loss = self.criterion(out, images) + self.kl_strength * kl_loss
                running_loss += loss.item()

        print(f'Loss of the network on test MNIST: {running_loss / len(self.test_dataloader)}%')
        print(f"Number of parameters: {self.model.get_num_params()}")

    def save_model(self, tag: str = ""):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.output_path / f"{tag}_model.pt")

    def sample(self, num_samples: int = 10, epoch: int = 0):
        with torch.no_grad():
            samples = self.model.sample(num_samples)
            samples = samples.cpu()
            # Save as images
            torchvision.utils.save_image(samples, self.output_path / f"epoch_{epoch}_samples.png", nrow=10)





