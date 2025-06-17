import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from collections import defaultdict
from torch.utils.data import Subset


def plot_point_cloud(point_cloud, title="Point Cloud"):
    """
    Visualize a 2D point cloud using matplotlib.
    Args:
        point_cloud (torch.Tensor): A tensor of shape (N, 2) representing the point cloud.
        title (str): Title of the plot.
    """
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1])
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


class PointCloudAugmentation:
    def __init__(self, rotation_range=50, scale_range=(0.5, 1.5), translation_range=0.3, jitter_std=0.002):
        self.rotation_range = math.radians(rotation_range)  # rotation range in radians
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.jitter_std = jitter_std

    def __call__(self, points):
        points = self.random_rotation(points)
        points = self.random_scale(points)
        points = self.random_translation(points)
        points = self.random_jitter(points)
        return points.float()

    def random_rotation(self, points):
        theta = torch.empty(1, device=points.device).uniform_(-self.rotation_range, self.rotation_range).item()
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rot_matrix = torch.tensor([[cos_theta, -sin_theta],
                                   [sin_theta, cos_theta]], dtype=points.dtype, device=points.device)
        return  rot_matrix @ points   # shape-safe: (N, 2) @ (2, 2)

    def random_scale(self, points):
        scale = torch.empty(1, device=points.device).uniform_(*self.scale_range).item()
        return points * scale

    def random_translation(self, points):
        translation = torch.empty((2, 1), dtype=points.dtype, device=points.device).uniform_(
            -self.translation_range, self.translation_range)
        return points + translation

    def random_jitter(self, points):
        jitter = torch.empty_like(points).normal_(mean=0.0, std=self.jitter_std)
        return points + jitter

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, augment_fn):
        self.base_dataset = base_dataset
        self.augment_fn = augment_fn

    def __getitem__(self, idx):
        points, label = self.base_dataset[idx]
        points_aug = self.augment_fn(points)
        return points_aug, label

    def __len__(self):
        return len(self.base_dataset)

def set_seed(seed,device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
      torch.cuda.manual_seed_all(seed)


def stratified_split(dataset, n_train_per_class=60, n_val_per_class=10, n_test_per_class=10, seed=42):
    np.random.seed(seed)

    # Check total per class
    total_per_class = n_train_per_class + n_val_per_class + n_test_per_class
    assert total_per_class == 80, f"Total per class must be 80, got {total_per_class}"

    # Collect indices for each class
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    for label, indices in class_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)

        assert len(indices) >= total_per_class, f"Not enough samples for class {label}"

        train_indices.extend(indices[:n_train_per_class])
        val_indices.extend(indices[n_train_per_class:n_train_per_class + n_val_per_class])
        test_indices.extend(indices[n_train_per_class + n_val_per_class:total_per_class])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset
