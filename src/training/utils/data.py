import torch.nn as nn
import torchvision

from src.constants import INPUT_DIR


def loss_factory(params):
    if params["dataset_name"] == "mnist":
        return torchvision.datasets.MNIST(
            root=INPUT_DIR,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ), torchvision.datasets.MNIST(
            root=INPUT_DIR,
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif params["dataset_name"] == "voc_segmentation":
        return torchvision.datasets.VOCSegmentation(
            root=INPUT_DIR,
            year="2012",
            image_set="train",
            download=True,
            transform=torchvision.transforms.ToTensor(),
            target_transform=torchvision.transforms.ToTensor(),
        ), torchvision.datasets.VOCSegmentation(
            root=INPUT_DIR,
            year="2012",
            image_set="val",
            download=True,
            transform=torchvision.transforms.ToTensor(),
            target_transform=torchvision.transforms.ToTensor(),
        )
    else:
        raise ValueError("Invalid dataset name")
