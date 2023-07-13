from model import model_factory
from optimizer import optimizer_factory
from training.classification_training_module import ClassificationTrainingModule
from training.vae_training_module import VAETrainingModule


def run(config):
    model = model_factory(config["model"])

    optimizer = optimizer_factory(model.parameters(), config["optimizer"])

    tm = None
    if config.get("mode") == "vae":
        tm = VAETrainingModule(model, optimizer, config["training"])
    elif config.get("mode") == "classification":
        tm = ClassificationTrainingModule(model, optimizer, config["training"])
    elif config.get("mode") == "segmentation":
        tm = ClassificationTrainingModule(model, optimizer, config["training"])
    else:
        raise ValueError("Invalid mode")

    tm.fit(num_epochs=config["training"]["n_epochs"])
    tm.eval()

    print(f"Saving model ...")
    tm.save_model()
