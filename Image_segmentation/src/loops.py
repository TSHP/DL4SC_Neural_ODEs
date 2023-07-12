
import torch
import torchvision
import torch.nn as nn
import numpy
from torch.nn import ReLU
from torch import optim
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, train_loader, test_loader, model, lossf, learning_rate, num_epochs, print_interval, filepath, device):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = lossf
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = num_epochs
        self.print_int = print_interval
        self.filepath = filepath
        self.device = device

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            eval_counter = 0
            for i, (inputs, labels) in enumerate(self.train_loader, 1):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                eval_counter += 1

                # Print the running loss every 'print_interval' evaluations
                if eval_counter % self.print_int == 0:
                    average_loss = running_loss / self.print_int
                    print(f"Epoch [{epoch+1}/{self.epochs}], Evaluation [{i}/{len(self.train_loader)}], Loss: {average_loss}")
                    running_loss = 0.0
                    eval_counter = 0

            # Print the remaining running loss at the end of the epoch
            if eval_counter > 0:
                average_loss = running_loss / eval_counter
                print(f"Epoch [{epoch+1}/{self.epochs}], Evaluation [{i}/{len(self.train_loader)}], Loss: {average_loss}")

    def test(self):
        self.model.eval()
        running_loss = 0.
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

        final_loss = running_loss / len(self.test_loader)
        print(f"Test avg dice loss: {final_loss:.2f}%")
        
    def predict(self, dataloader=None):
        if dataloader is None:
            dataloader = self.test_loader
        self.model.eval()
        predictions_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = self.model(inputs)
                predictions_list.append(outputs.numpy())
                labels_list.append(labels.numpy())

        return predictions_list, labels_list
    
    def save_model(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        torch.save(self.model.state_dict(), filepath)
        print("Model saved successfully at " + str(filepath))

    def load_model(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        self.model.load_state_dict(torch.load(filepath))
        print("Model loaded successfully from " + str(filepath))
