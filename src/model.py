from src.network.slmlp import SLMLP
from src.network.resnet import ResNet
from src.network.rknet import RKNet
from src.network.odenet import ODENet
from src.network.neural_ae import NeuralAE


def model_factory(params):
    if params["network_name"] == "slmlp":
        return SLMLP(
            in_dim=params["in_dim"],
            out_dim=params["out_dim"],
            hidden_dim=params["hidden_dim"]
        )
    elif params["network_name"] == "resnet":
        return ResNet(
            out_dim=params["out_dim"],
        )
    elif params["network_name"] == "rknet":
        return RKNet(
            out_dim=params["out_dim"]
        )
    elif params["network_name"] == "odenet":
        return ODENet(
            out_dim=params["out_dim"]
        )
    elif params["network_name"] == "nae":
        return NeuralAE(
            out_dim=params["out_dim"]
        )
    else:
        raise ValueError("Invalid network name")
