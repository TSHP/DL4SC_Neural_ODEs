import torch


def dice_loss(predicted, target):
    smooth = 1e-7  # A small constant to avoid division by zero
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss = 1.0 - dice
    return loss
