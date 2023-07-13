import torch.nn as nn


def loss_factory(params):
    if params["loss_name"] == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss name")
