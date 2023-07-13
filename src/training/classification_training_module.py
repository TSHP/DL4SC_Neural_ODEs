import torch

from training.abc_training_module import ABCTrainingModule


class ClassificationTrainingModule(ABCTrainingModule):
    def __init__(self, model, optimizer, params) -> None:
        self.super().__init__(model, optimizer, params)     

    def train_step(self, images, labels):
        out = self.model(images)
        loss = self.criterion(out, labels)
        step_loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return step_loss
    
    def eval_step(self, images, labels):
        out = self.model(images)
        step_loss = self.criterion(out, labels)
        return step_loss
    
    def test(self):
        total_incorrect = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for _, (images, labels) in enumerate(self.test_dataloader):
                out = self.model(images)
                loss = self.criterion(out, labels)
                running_loss += loss.item()

                _, predicted = torch.max(out.data, 1)
                total_incorrect += (predicted != labels).sum().item()
                total += labels.size(0)

        print(f"Error of the network on test: {100 * total_incorrect / total}%")
        print(
            f"Loss of the network on test: {running_loss / len(self.test_dataloader)}%"
        )
        print(f"Number of parameters: {self.model.get_num_params()}")



