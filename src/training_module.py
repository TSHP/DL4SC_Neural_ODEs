from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision

from src.constants import INPUT_DIR


class TrainingModule:
    def __init__(self, model, optimizer, params) -> None:
        self.model = model
        self.optimizer = optimizer

        self.batch_size = params["batch_size"]

        self.training_dataset = torchvision.datasets.MNIST(root=INPUT_DIR, train=True,
                                    download=True, transform=torchvision.transforms.ToTensor()) # Might need to normalize
        self.training_dataloader = torch.utils.data.DataLoader(self.training_dataset, batch_size=self.batch_size,
                                      shuffle=True)
        
        self.training_dataset = torchvision.datasets.MNIST(root=INPUT_DIR, train=True,
                                    download=True, transform=torchvision.transforms.ToTensor()) # Might need to normalize
        self.training_dataloader = torch.utils.data.DataLoader(self.training_dataset, batch_size=self.batch_size,
                                      shuffle=True)
        
        self.test_dataset = torchvision.datasets.MNIST(root=INPUT_DIR, train=False,
                                    download=True, transform=torchvision.transforms.ToTensor()) # Might need to normalize
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                      shuffle=False)

        self.criterion = nn.CrossEntropyLoss()

    def fit(self, num_epochs: int = 10):
        total_steps = len(self.training_dataloader)
        for cur_epoch in (pbar_epoch := tqdm(range(num_epochs))):
            running_loss = 0.0
            for i, (images, labels) in (pbar := tqdm(enumerate(self.training_dataloader))):
                images = images.reshape(-1, 28*28)
                out = self.model(images)
                loss = self.criterion(out, labels)
                running_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_description(f'Epoch[{cur_epoch + 1}/{num_epochs}], Step[{i + 1}/{total_steps}], Losses: {loss.item():.4f}') 
            pbar_epoch.set_description(f'Epoch[{cur_epoch + 1}/{num_epochs}], Running Loss: {running_loss:.4f}')      

    def eval(self):
        total_correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_dataloader):
                images = images.reshape(-1, 28*28)
                out = self.model(images)
                loss = self.criterion(out, labels)

                _, predicted = torch.max(out.data, 1)
                total_correct += (predicted == labels).sum().item()
                total += labels.size(0)


        print(f'Accuracy of the network on test MNIST: {100 * total_correct / total}%')


