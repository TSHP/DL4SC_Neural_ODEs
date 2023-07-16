from pathlib import Path

import torch
from torchvision.utils import make_grid
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def color_mask(mask: np.ndarray):
    """Transforms the masks from greyscale to rgb for visualization of the predictions."""
    label_colors = np.array([(0, 0, 0),
                             (255, 51, 51), (179, 81, 76), (161, 161, 0), (153, 153, 153), (2, 255, 255),
                             (0, 50, 0), (255, 102, 255), (51, 51, 255), (52, 255, 50), (0, 0, 153), # Last one here, colours might be bad
                             (0, 255, 255), (255, 255, 0), (255, 0, 255), (255, 0, 0), (0, 255, 0),
                             (0, 0, 255), (255, 255, 2), (0, 161, 161), (255, 51, 51), (255, 255, 255)])
    rgb = np.zeros(mask.shape + (3,)).astype(np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            label = int(mask[i, j])
            rgb[i, j] = label_colors[label]
    return rgb


def vizualize_segmentation_predictions(save_path: Path, images: np.ndarray, gts: np.ndarray, predictions: np.ndarray):
    """Plots image, ground truth, predicted mask"""
    nrow = 3
    ncol = len(images)

    _ = plt.figure(figsize=(ncol+1, nrow+1)) 
    
    plt.figure(figsize=(50, 30))
    gs1 = gridspec.GridSpec(nrow+1, ncol, height_ratios=[2, 2, 2, 0.25])
    gs1.update(wspace=0.0, hspace=0.05, top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
         left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 

    for i, image in enumerate(images):
        ax1 = plt.subplot(gs1[0, i], xticks=[], yticks=[])
        ax1.set_aspect("auto")
        ax1.imshow(np.transpose(image, (1, 2, 0)))

    for i, gt in enumerate(gts):
        ax1 = plt.subplot(gs1[1, i], xticks=[], yticks=[])
        ax1.set_aspect("auto")
        ax1.imshow(color_mask(gt))

    for i, prediction in enumerate(predictions):
        ax1 = plt.subplot(gs1[2, i], xticks=[], yticks=[])
        ax1.set_aspect("auto")
        ax1.imshow(color_mask(prediction))


    plt.tight_layout()
    plt.savefig(save_path)


def visualize_results(save_path, predicts, masks):
    
    predicts = torch.tensor(predicts[0])
    masks = torch.tensor(masks[0])
    
    predicts = torch.argmax(predicts, dim=1, keepdim=True)
    masks = torch.argmax(masks, dim=1, keepdim=True)

    grid_images = make_grid(predicts)
    grid_masks = make_grid(masks)

    # Convert tensors to numpy arrays
    grid_images_np = grid_images[0,:,:].numpy()
    grid_masks_np = grid_masks[0,:,:].numpy()

    # Display the grid of images and masks
    plt.subplot(1, 2, 1)
    plt.imshow(grid_images_np, cmap="jet",vmin=0, vmax=20)
    plt.title("Predictions")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(grid_masks_np, cmap="jet",vmin=0, vmax=20)
    plt.title("Masks")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
