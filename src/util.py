import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import random
from collections import defaultdict
from torch.utils.data import Subset
import numpy as np


def plot_point_cloud(point_cloud, title="Point Cloud"):
    """
    Visualize a 2D point cloud using matplotlib.
    Args:
        point_cloud (torch.Tensor): A tensor of shape (N, 2) representing the point cloud.
        title (str): Title of the plot.
    """

    if point_cloud.shape[0] == 2 and point_cloud.shape[1] > 2:
        point_cloud = point_cloud.T
    elif point_cloud.shape[1] != 2:
        raise ValueError(f"point_cloud must have shape (N,2) or (2,N), got {point_cloud.shape}")

    plt.scatter(point_cloud[:, 0], point_cloud[:, 1])
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


class PointCloudAugmentation:
    def __init__(self, rotation_range=90, scale_range=(0.5, 1.5), translation_range=0.2, jitter_std=0.008, max_dropout_ratio=0.45):
        self.rotation_range = math.radians(rotation_range)  # rotation range in radians
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.jitter_std = jitter_std
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        points = self.random_rotation(points)
        points = self.random_scale(points)
        points = self.random_translation(points)
        points = self.random_jitter(points)
        points = self.random_point_dropout(points)
        return points.float()

    def random_rotation(self, points):
        theta = torch.empty(1, device=points.device).uniform_(-self.rotation_range, self.rotation_range).item()
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rot_matrix = torch.tensor([[cos_theta, -sin_theta],
                                   [sin_theta, cos_theta]], dtype=points.dtype, device=points.device)
        return  rot_matrix @ points

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
        
    def random_point_dropout(self, points, max_dropout_ratio=0.2):

        dropout_ratio = torch.empty(1).uniform_(0, max_dropout_ratio).item()
        num_points = points.shape[1]
        num_drop = int(num_points * dropout_ratio)
        
        if num_drop > 0:
            indices = torch.randperm(num_points, device=points.device)[:num_drop]
            points[:, indices] = 0
        return points

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, img_transform, point_augment_fn=None):
        self.base_dataset = base_dataset
        self.img_transform = img_transform
        self.point_augment_fn = point_augment_fn

    def __getitem__(self, idx):
        points, label, image = self.base_dataset[idx]

        if self.point_augment_fn:
            points = self.point_augment_fn(points)

        if self.img_transform:
            image = self.img_transform(image)
        
        return points, label, image

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

    total_per_class = n_train_per_class + n_val_per_class + n_test_per_class
    assert total_per_class == 80, f"Total per class must be 80, got {total_per_class}"

    class_indices = defaultdict(list)
    for idx in range(len(dataset)):

        _, label, _ = dataset[idx] 
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    for label, indices in class_indices.items():
        indices = np.array(indices)
        np.random.shuffle(indices)

        if len(indices) < total_per_class:
            print(f"Warning: Class {label} has only {len(indices)} samples (expected {total_per_class}).")
            
        train_indices.extend(indices[:n_train_per_class])
        val_indices.extend(indices[n_train_per_class:n_train_per_class + n_val_per_class])
        test_indices.extend(indices[n_train_per_class + n_val_per_class:total_per_class])

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset
    
    
    
def accuracy_topk(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1) 
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # Check if ANY of the top k were correct
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        
