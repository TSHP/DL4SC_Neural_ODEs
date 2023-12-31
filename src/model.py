from src.network.classification.mlp import MLP
from src.network.classification.resnet import ResNet
from src.network.classification.odenet import ODENet
from src.network.generative.neural_vae import NeuralVAE
from src.network.segmentation.res_unet import ResUNet
from src.network.segmentation.ode_unet import ODEUNet
from src.network.segmentation.odenet_segmentation import SegmentationODENet
from src.network.segmentation.resnet_segmentation import SegmentationResNet


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
            num_res_blocks=params["num_res_blocks"],
            num_filters=params["num_filters"],
            kernel_size=params["kernel_size"],
            num_channels=params["num_channels"],
            out_dim=params["out_dim"],
        )
    # ODE-Net
    elif params["network_name"] == "odenet":
        return ODENet(out_dim=params["out_dim"])
    # VAE
    elif params["network_name"] == "nvae":
        return NeuralVAE(
            params["latent_dim"],
            adjoint=params.get("adjoint", False),
            rtol=params["rtol"],
            atol=params["atol"],
            method=params["method"],
        )
    # U-Net with residual blocks for segmentation
    elif params["network_name"] == "resunet":
        return ResUNet(
            num_filters=params["num_filters"],
            kernel_size=params["kernel_size"],
            out_dim=params["out_dim"],
        )
    # U-Net with ODE blocks for segmentation
    elif params["network_name"] == "odeunet":
        return ODEUNet(
            num_filters=params["num_filters"],
            kernel_size=params["kernel_size"],
            out_dim=params["out_dim"],
            adjoint=params["adjoint"],
            rtol=params["rtol"],
            atol=params["atol"],
            method=params["method"],
        )
    # ODE-Net for segmentation
    elif params["network_name"] == "odenet_segmentation":
        return SegmentationODENet(
            num_filters=params["num_filters"],
            kernel_size=params["kernel_size"],
            out_dim=params["out_dim"],
            adjoint=params["adjoint"],
            rtol=params["rtol"],
            atol=params["atol"],
            method=params["method"],
        )
    # ResNet for segmentation
    elif params["network_name"] == "resnet_segmentation":
        return SegmentationResNet(
            num_filters=params["num_filters"],
            kernel_size=params["kernel_size"],
            out_dim=params["out_dim"],
        )
    else:
        raise ValueError("Invalid network name")
