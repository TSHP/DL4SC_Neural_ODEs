from src.network.slmlp import SLMLP
from src.network.resnet import ResNet
from src.network.odenet import ODENet
from network.generative.neural_vae import NeuralVAE


def model_factory(params):
    if params["network_name"] == "slmlp":
        return SLMLP(
            in_dim=params["in_dim"],
            out_dim=params["out_dim"],
            hidden_dim=params["hidden_dim"],
        )
    elif params["network_name"] == "resnet":
        return ResNet(
            out_dim=params["out_dim"],
        )
    elif params["network_name"] == "odenet":
        return ODENet(out_dim=params["out_dim"])
    elif params["network_name"] == "nvae":
        return NeuralVAE(params["latent_dim"])
    else:
        raise ValueError("Invalid network name")
