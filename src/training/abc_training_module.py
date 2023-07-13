from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np

from training.utils.loss import loss_factory
from training.utils.data import dataset_factory
from constants import MODEL_DIR


class ABCTrainingModule(ABC):
    def __init__(self, model, optimizer, params) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = loss_factory(params)

        self.batch_size = params["batch_size"]

        # Load dataset
        self.dataset, self.test_dataset = dataset_factory(params)

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [0.8, 0.2]
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Setup output directory
        self.output_path = Path(params["output_path"])
        self.output_path.mkdir(parents=True, exist_ok=True)

        #Set device
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        print("Using device:", self.device)
        self.model.to(self.device)

    def fit(self, num_epochs: int = 100):
        best_val_loss = float("inf")
        train_loss_history = []
        val_loss_history = []
        val_metrics_history = []
        for cur_epoch in (pbar_epoch := tqdm(range(num_epochs))):
            running_loss = 0.0
            for _, (images, labels) in tqdm(enumerate(self.train_dataloader)):
                loss = self.train_step(images, labels)
                running_loss += loss
                train_loss_history.append(loss)

            running_val_loss = 0.0
            val_predictions = []
            with torch.no_grad():
                for _, (images, labels) in tqdm(enumerate(self.val_dataloader)):
                    out, loss = self.eval_step(images, labels)
                    running_val_loss += loss
                    val_predictions.append(out)

                val_loss_history.append(running_val_loss)       

            # Show metrics in pbar
            pbar_description = f"Epoch[{cur_epoch + 1}/{num_epochs}], Loss: {running_loss / len(self.train_dataloader):.4f}, Val Loss: {running_val_loss / len(self.val_dataloader):.4f}"
            val_metrics = self.metrics(torch.stack(val_predictions, 0), labels)
            val_metrics_history.append(val_metrics)
            for k, v in val_metrics.items():
                pbar_description += f", Val {k}: {v}"
            pbar_epoch.set_description(
                pbar_description
            )

            if running_val_loss < best_val_loss:
                best_val_loss = running_val_loss
                self.save_model("best_val")

        # Save histories as numpy arrays
        np.save(self.output_path / "train_loss_history.npy", np.array(train_loss_history))
        np.save(self.output_path / "val_loss_history.npy", np.array(val_loss_history))
        np.save(self.output_path / "val_metrics_history.npy", np.array(val_metrics_history))

        self.save_model("last")

    def test(self, model_tag):
        """Test the model and save the results"""
        self.model.load_state_dict(torch.load(self.output_path / f"{model_tag}_model.pt"))
        self.model.eval()
        running_test_loss = 0.0
        test_predictions = []

        with torch.no_grad():
            for _, (images, labels) in tqdm(enumerate(self.test_dataloader)):
                out, loss = self.eval_step(images, labels)
                running_test_loss += loss
                test_predictions.append(out)

    def save_model(self, tag: str = "last"):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.output_path / f"{tag}_model.pt")

    @abstractmethod
    def compute_loss(self, inputs, labels):
        """Returns loss"""
        pass

    @abstractmethod
    def train_step(self, images, labels):
        """Returns loss"""
        pass
    
    @abstractmethod
    def eval_step(self, images, labels):
        """Returns (model_prediction, loss)"""
        pass

    @abstractmethod
    def metrics(self, model_predictions, labels):
        """Returns a dictionary of metrics, the key will be used as teh display name in the progress bar"""
        pass
