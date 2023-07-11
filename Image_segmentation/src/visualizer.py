import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import torch

def visualize_dataset(voc_dataset, num=5):
    # Iterate over the first 5 samples in the dataset
    for i in range(num):
        # Get the image and mask from the dataset
        image, mask = voc_dataset[i]

        # Convert the image tensor to a PIL image and then to a NumPy array
        image_np = np.array(image)

        # Convert the mask tensor to a NumPy array
        mask_np = np.array(mask)

        # Display the image and mask
        plt.subplot(5, 2, i * 2 + 1)
        plt.imshow(image_np)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(5, 2, i * 2 + 2)
        plt.imshow(mask_np, cmap='jet', vmin=0, vmax=21)  # Assuming 21 classes in the segmentation mask
        plt.title("Segmentation Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_dataloader(VOC_data_loader):
        # Get the first batch from the data loader
    batch = next(iter(VOC_data_loader))

    # Extract images and masks from the batch
    images, masks = batch

    # Create a grid of images for visualization
    grid_images = make_grid(images)
    grid_masks = make_grid(masks)

    # Convert tensors to numpy arrays
    grid_images_np = grid_images.numpy().transpose(1, 2, 0)
    grid_masks_np = grid_masks.numpy().transpose(1, 2, 0)

    # Display the grid of images and masks
    plt.subplot(1, 2, 1)
    plt.imshow(grid_images_np)
    plt.title("Images")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(grid_masks_np[:, :, 0])
    plt.title("Masks")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    print("images shape: ", images.shape)
    print("masks shape: ", masks.shape)

def visualize_results(predicts, masks):
    
    predicts = torch.tensor(predicts[0])
    masks = torch.tensor(masks[0])
    
    # Create a grid of images for visualization
    grid_images = make_grid(predicts)
    grid_masks = make_grid(masks)

    # Convert tensors to numpy arrays
    grid_images_np = grid_images.numpy().transpose(1, 2, 0)
    grid_masks_np = grid_masks.numpy().transpose(1, 2, 0)

    # Display the grid of images and masks
    plt.subplot(1, 2, 1)
    plt.imshow(grid_images_np[:, :, 0])
    plt.title("Predictions")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(grid_masks_np[:, :, 0])
    plt.title("Masks")
    plt.axis("off")

    plt.tight_layout()
    plt.show()