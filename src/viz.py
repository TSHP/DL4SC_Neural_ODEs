from pathlib import Path

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