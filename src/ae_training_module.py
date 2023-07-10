from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision

from src.constants import INPUT_DIR


class AETrainingModule:
    def __init__(self, model, optimizer, params) -> None:
        self.model = model
        self.optimizer = optimizer

        self.batch_size = params["batch_size"]

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

    def fit(self, num_epochs: int = 100):
        total_steps = len(self.training_dataloader)
        for cur_epoch in (pbar_epoch := tqdm(range(num_epochs))):
            running_loss = 0.0
            for i, (images, _) in (pbar := tqdm(enumerate(self.training_dataloader))):
                out = self.model(images)
                loss = self.criterion(out, images)
                running_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_description(f'Epoch[{cur_epoch + 1}/{num_epochs}], Step[{i + 1}/{total_steps}], Loss: {loss.item():.4f}') 
            pbar_epoch.set_description(f'Epoch[{cur_epoch + 1}/{num_epochs}], Running Loss: {running_loss / len(self.training_dataloader):.4f}')      

    def eval(self):
        running_loss = 0.0
        with torch.no_grad():
            for _, (images, _) in enumerate(self.test_dataloader):
                out = self.model(images)
                loss = self.criterion(out, images)
                running_loss += loss.item()

        print(f'Loss of the network on test MNIST: {running_loss / len(self.test_dataloader)}%')
        print(f"Number of parameters: {self.model.get_num_params()}")



