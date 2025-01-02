#%%
import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from PIL import Image  # เพิ่ม import

class CoralWeaklySupervisedDataset(Dataset):
    def __init__(self, image_dir, segment_dir, output_size=(256, 256), transform=None, labels_use = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]):
        """
        Initializes the dataset by loading paths from a txt file and filtering based on coral_count.

        Args:
            txt_file (str): Path to the txt file containing image, mask, and segment file paths.
            transform (callable, optional): Optional transform to apply to images.
        """
        self.segment_dir = segment_dir
        self.image_dir = image_dir

        self.transform = transform
        self.labels_use = labels_use

        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        
        self.output_size = output_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get file paths
        file_data = self.data[idx]
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        segment_path = os.path.join(self.segment_dir, self.image_files[idx][:-3] + 'tif')

        # Load image, segmentation, and mask
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_LINEAR)

        if image is None:
            raise ValueError(f"Cannot load image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segmentation = cv2.imread(segment_path, cv2.IMREAD_UNCHANGED)
        segmentation = cv2.resize(segmentation, self.output_size, interpolation=cv2.INTER_NEAREST)

        if segmentation is None:
            raise ValueError(f"Cannot load segmentation at {segment_path}")

        mask = (segmentation > 0)

        # Process segmentation: Keep only coral class (label=1)
        segmentation = np.isin(segmentation, self.labels_use).astype(np.float32)  # Coral only
        mask = (mask >= 1).astype(np.float32)  # Binary mask

        # print("mask_count", mask.sum())
        # print("segmentation_count", segmentation.sum())

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32)  # HWC to CHW
        segmentation = torch.tensor(segmentation, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, segmentation, mask, file_data["image_filename"]

def create_color_segment(image, segmented_mask):
    label_map = {0: "None", 1: "Agalg", 2: "DCP", 3: "ROC", 4: "TWS", 5: "CCA", 6: "Davi", 7: "Ana", 8: "Other"}
    color_b = [0,   0,   0, 255,   0, 255, 255,   255, 128]
    color_g = [0,   0, 255,   0, 255,   0, 255,   255, 128]
    color_r = [0, 255,   0,   0, 255, 255,   0,   255, 128]
    img = np.zeros_like(image)
    for i in range(9):
        if i == 0:
            continue
        mask = segmented_mask == i
        img[mask] = (color_b[i], color_g[i], color_r[i])
    return img

# Inverse normalization
def inverse_normalize(image, mean, std):
    """
    Reverses normalization on an image tensor.

    Args:
        image (torch.Tensor): Normalized image tensor of shape [C, H, W].
        mean (list): Mean used for normalization (per channel).
        std (list): Std used for normalization (per channel).

    Returns:
        torch.Tensor: Image tensor with normalization reversed.
    """
    mean = torch.tensor(mean).view(-1, 1, 1)  # Reshape mean to [C, 1, 1]
    std = torch.tensor(std).view(-1, 1, 1)    # Reshape std to [C, 1, 1]
    return image * std + mean
    
def show_image_with_predictions(
    image, segmentation_mask, predicted_mask,
    epoch=0, image_num=0, save_file=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], point_size=30
):
    """
    Display or save ground truth and predicted segmentation results side-by-side.

    Args:
        image (torch.Tensor): Normalized image tensor with shape [3, H, W].
        segmentation_mask (torch.Tensor): Ground truth segmentation mask with shape [H, W].
        points_mask (torch.Tensor): Ground truth points mask, shape [H, W].
        predicted_mask (torch.Tensor): Predicted segmentation mask with shape [H, W].
        predicted_points_mask (torch.Tensor): Predicted points mask, shape [H, W].
        epoch (int): Current epoch number.
        image_num (int): Current image number.
        save_file (bool): Whether to save the image instead of displaying it.
        mean (list): Mean used for normalization (per channel).
        std (list): Std used for normalization (per channel).
        point_size (int): Size of the points in the plot.
    """
    # Inverse normalize the image
    image = inverse_normalize(image, mean, std)

    # Convert image tensor to numpy array and clip to [0, 1] for display
    image = image.permute(1, 2, 0).cpu().numpy()  # Convert to [H, W, C]
    image = np.clip(image, 0, 1)  # Ensure values are in [0, 1]

    # Define label map and colors
    label_map = {0: "None", 1: "Agalg", 2: "DCP", 3: "ROC", 4: "TWS", 5: "CCA", 6: "Davi", 7: "Ana", 8: "Other"}
    label_colors = {
        0: "black", 1: "red", 2: "green", 3: "blue", 4: "yellow", 5: "purple",
        6: "orange", 7: "cyan", 8: "white"
    }
    custom_cmap = ListedColormap([label_colors[key] for key in sorted(label_map.keys())])

    # Create a folder for saving results if save_file=True
    if save_file:
        save_dir = f"./train_results/epoch_{epoch}/"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"image{image_num}.jpg")

    # Create a figure for ground truth and prediction
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # Ground truth visualization
    axes[0].imshow(image)
    if segmentation_mask is not None:
        axes[0].imshow(segmentation_mask.cpu().numpy(), alpha=0.5, cmap=custom_cmap)
    if segmentation_mask is not None:
        y, x = torch.where(segmentation_mask > 0)
        labels = segmentation_mask[y, x].cpu().numpy()
        for unique_label in np.unique(labels):
            idx = labels == unique_label
            axes[0].scatter(
                x[idx], y[idx], c=label_colors.get(unique_label, "black"),
                s=point_size, label=label_map.get(unique_label, "Unknown")
            )
    axes[0].axis("off")
    axes[0].set_title("Ground Truth")

    # Prediction visualization
    # axes[1].imshow(image)
    if predicted_mask is not None:
        segmented_color = create_color_segment(image, predicted_mask.cpu().numpy())
        axes[1].imshow(segmented_color)
    if predicted_mask is not None:
        y, x = torch.where(predicted_mask > 0)
        labels = predicted_mask[y, x].cpu().numpy()
        for unique_label in np.unique(labels):
            idx = labels == unique_label
            axes[1].scatter(
                x[idx], y[idx], c=label_colors.get(unique_label, "black"),
                s=point_size, label=label_map.get(unique_label, "Unknown")
            )
    axes[1].axis("off")
    axes[1].set_title("Prediction")

    # Save or show the figure
    if save_file:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

#%% 
dataset_path = "./dataset/"
image_dir = dataset_path + "images"
segment_dir = dataset_path + "segmentations"
mask_dir = dataset_path + "masks"

dataset = CoralWeaklySupervisedDataset(
    image_dir=image_dir,
    segment_dir=segment_dir,
    transform=None
)

print(f"Filtered dataset size: {len(dataset)}")
