import torch.nn as nn

from src.training.utils.kl_divergence import KLDivLoss


def loss_factory(params):
    losses = []
    for loss_name in params["loss_names"]:
        losses.append(construct_loss(loss_name, params))
    
def construct_loss(loss_name):
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    if loss_name == "kl":
        return KLDivLoss()
    else:
        raise ValueError("Invalid loss name")
    
