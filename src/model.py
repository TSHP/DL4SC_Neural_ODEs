from src.network.classification.mlp import MLP
from src.network.classification.resnet import ResNet
from src.network.classification.odenet import ODENet
from src.network.generative.neural_vae import NeuralVAE


def model_factory(params):
    if params["network_name"] == "mlp":
        return MLP(
            in_dim=params["in_dim"],
            out_dim=params["out_dim"],
            hidden_dims=params["hidden_dims"],
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
