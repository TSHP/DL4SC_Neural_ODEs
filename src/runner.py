from src.model import model_factory
from src.optimizer import optimizer_factory
from src.training_module import TrainingModule


def run(config):
    model = model_factory(config["model"])

    optimizer = optimizer_factory(
        model.parameters(), config["optimizer"]
    )

    tm = TrainingModule(model, optimizer, config["training"])

    tm.fit(num_epochs=config["training"]["n_epochs"])

    tm.eval()
