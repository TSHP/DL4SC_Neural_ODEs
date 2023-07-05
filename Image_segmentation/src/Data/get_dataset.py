
from torchvision.datasets import VOCSegmentation
import os
import pathlib

def download_VOC():
    # Set the root directory where you want to download the dataset
    root = pathlib.Path(__file__).parent.resolve()
    root = root / "data_set"
    year = "2012"

    # Download the VOC dataset
    voc_trainset = VOCSegmentation(root=root, year=year, image_set="train", download=True)
    voc_valset = VOCSegmentation(root=root, year=year, image_set="val", download=True)

    return voc_trainset, voc_valset