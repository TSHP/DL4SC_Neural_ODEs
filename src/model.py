from src.network.mlp import MLP
from src.network.resnet import ResNet
from src.network.rknet import RKNet
from src.network.odenet import ODENet


def model_factory(params):
    if params["network_name"] == "mlp":
        return MLP(
            in_dim=params["in_dim"],
            out_dim=params["out_dim"]
        )
    elif params["network_name"] == "resnet":
        return ResNet(
            in_dim=params["in_dim"],
            out_dim=params["out_dim"],
            hidden_dim=params["hidden_dim"],
            num_hidden=params["num_hidden"],
        )
    elif params["network_name"] == "rknet":
        return RKNet(
            in_dim=params["in_dim"],
            out_dim=params["out_dim"],
            hidden_dim=params["hidden_dim"],
            num_hidden=params["num_hidden"],
        )
    elif params["network_name"] == "odenet":
        return ODENet(
            in_dim=params["in_dim"],
            out_dim=params["out_dim"],
            hidden_dim=params["hidden_dim"],
            num_hidden=params["num_hidden"],
        )
    else:
        raise ValueError("Invalid network name")
