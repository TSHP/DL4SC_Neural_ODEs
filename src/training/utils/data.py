import torch
import torch.nn.functional as F

import numpy as np

import torchvision
import torchvision.transforms as transforms

from src.constants import INPUT_DIR

def one_hot(x):
    encoded = np.asarray(x).copy()
    encoded[encoded == 255] = 0
    encoded = F.one_hot(torch.tensor(encoded).to(torch.int64), 21).permute(2, 0, 1).squeeze().to(torch.float)
    #reshaped = encoded.transpose(0, 3).to(torch.float).squeeze(-1)
    return encoded

def dataset_factory(params):
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
        img_transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )

        mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (128, 128), interpolation=transforms.InterpolationMode.NEAREST
                ),
                lambda x: one_hot(x)
            ]
        )

        return torchvision.datasets.VOCSegmentation(
            root=INPUT_DIR,
            year="2012",
            image_set="train",
            download=True,
            transform=img_transform,
            target_transform=mask_transform,
        ), torchvision.datasets.VOCSegmentation(
            root=INPUT_DIR,
            year="2012",
            image_set="val",
            download=True,
            transform=img_transform,
            target_transform=mask_transform,
        )
    else:
        raise ValueError("Invalid dataset name")
