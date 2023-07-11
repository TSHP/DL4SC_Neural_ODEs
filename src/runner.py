from src.model import model_factory
from src.optimizer import optimizer_factory
from src.training_module import TrainingModule
from src.ae_training_module import AETrainingModule


def run(config):
    model = model_factory(config["model"])

    optimizer = optimizer_factory(
        model.parameters(), config["optimizer"]
    )

    tm = None
    if config.get("mode") == "autoencoder":
        tm = AETrainingModule(model, optimizer, config["training"])
    else:
        tm = TrainingModule(model, optimizer, config["training"])

    tm.fit(num_epochs=config["training"]["n_epochs"])
    tm.eval()

    print(f"Saving model ...")
    tm.save_model()
