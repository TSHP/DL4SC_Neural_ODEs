import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from src.constants import INPUT_DIR


def one_hot(x):
    x[x == 255] = 0
    x = F.one_hot(x.to(torch.int64), 21).permute(0, 3, 2, 1).to(torch.float)
    x = torch.squeeze(x)
    return x


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
                transforms.ToTensor(),
                lambda x: one_hot(x),
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
