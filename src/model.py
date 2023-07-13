from src.network.classification.mlp import MLP
from src.network.classification.resnet import ResNet
from src.network.classification.odenet import ODENet
from src.network.generative.neural_vae import NeuralVAE
from src.network.segmentation.resnet import ResNet as ResNetSegmentation
from src.network.segmentation.odenet import ODENet as ODENetSegmentation


def model_factory(params):
    # MLPs
    if params["network_name"] == "mlp":
        return MLP(
            in_dim=params["in_dim"],
            out_dim=params["out_dim"],
            hidden_dims=params["hidden_dims"],
        )
    # ResNet
    elif params["network_name"] == "resnet":
        return ResNet(
            out_dim=params["out_dim"],
            num_res_blocks=params["num_res_blocks"],
        )
    # ODE-Net
    elif params["network_name"] == "odenet":
        return ODENet(out_dim=params["out_dim"])
    # VAE
    elif params["network_name"] == "nvae":
        return NeuralVAE(
            params["latent_dim"],
            adjoint=params["adjoint"],
            rtol=params["rtol"],
            atol=params["atol"],
            method=params["method"],
        )
    # ResNet for segmentation
    elif params["network_name"] == "resnet_segmentation":
        return ResNetSegmentation(
            out_dim=params["out_dim"],
            num_res_blocks=params["num_res_blocks"],
        )
    # ODE-Net for segmentation
    elif params["network_name"] == "odenet_segmentation":
        return ODENetSegmentation(
            params["out_dim"],
            adjoint=params["adjoint"],
            rtol=params["rtol"],
            atol=params["atol"],
            method=params["method"],
        )
    else:
        raise ValueError("Invalid network name")
