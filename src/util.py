"""
Utility functions: augmentation, dataset wrappers, data splitting, seeding, metrics.
"""

import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from collections import defaultdict
from torch.utils.data import Subset
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import Dataset




def plot_point_cloud(point_cloud, title="Point Cloud"):
    """Visualize a 2D point cloud using matplotlib.

    Args:
        point_cloud: Tensor of shape (N, 2) or (2, N).
        title: Plot title.
    """
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.numpy()

    if point_cloud.shape[0] == 2 and point_cloud.shape[1] > 2:
        point_cloud = point_cloud.T
    elif point_cloud.shape[1] != 2:
        raise ValueError(f"point_cloud must have shape (N,2) or (2,N), got {point_cloud.shape}")

    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], s=5, alpha=0.5)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.show()


# ---------------------------------------------------------------------------
# Point Cloud Augmentation
# ---------------------------------------------------------------------------

class PointCloudAugmentation:
    """Augmentation pipeline for 2D point clouds (shape: 2 x N, channels-first).

    Applies: rotation → scale → translation → jitter → point dropout.
    """

    def __init__(self, rotation_range=90, scale_range=(0.5, 1.5),
                 translation_range=0.2, jitter_std=0.008, point_dropout_rate=0.2):
        self.rotation_range = math.radians(rotation_range)
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.jitter_std = jitter_std
        self.point_dropout_rate = point_dropout_rate

    def __call__(self, points):
        points = self.random_rotation(points)
        points = self.random_scale(points)
        points = self.random_translation(points)
        points = self.random_jitter(points)
        points = self.random_point_dropout(points)
        return points.float()

    def random_rotation(self, points):
        theta = random.uniform(-self.rotation_range, self.rotation_range)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rot_matrix = torch.tensor(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
            dtype=points.dtype, device=points.device,
        )
        return rot_matrix @ points

    def random_scale(self, points):
        scale = random.uniform(*self.scale_range)
        return points * scale

    def random_translation(self, points):
        translation = torch.empty(
            (2, 1), dtype=points.dtype, device=points.device
        ).uniform_(-self.translation_range, self.translation_range)
        return points + translation

    def random_jitter(self, points):
        jitter = torch.empty_like(points).normal_(mean=0.0, std=self.jitter_std)
        return points + jitter

    def random_point_dropout(self, points):
        """Randomly zero out up to point_dropout_rate fraction of points."""
        dropout_ratio = random.uniform(0, self.point_dropout_rate)
        num_points = points.shape[1]
        num_drop = int(num_points * dropout_ratio)

        if num_drop > 0:
            indices = torch.randperm(num_points, device=points.device)[:num_drop]
            points[:, indices] = 0
        return points


# ---------------------------------------------------------------------------
# Augmented Dataset Wrapper
# ---------------------------------------------------------------------------

class AugmentedDataset(torch.utils.data.Dataset):
    """Wraps a dataset and applies augmentations on-the-fly.

    Supports both point-only datasets (2-tuple) and fusion datasets (3-tuple).

    Args:
        base_dataset: Underlying dataset returning (points, label) or (points, label, image).
        img_transform: Optional torchvision transform for images.
        point_augment_fn: Optional callable for point cloud augmentation (None = no augmentation).
    """

    def __init__(self, base_dataset, img_transform=None, point_augment_fn=None):
        self.base_dataset = base_dataset
        self.img_transform = img_transform
        self.point_augment_fn = point_augment_fn

    def __getitem__(self, idx):
        item = self.base_dataset[idx]

        if len(item) == 3:
            points, label, image = item
            if self.point_augment_fn:
                points = self.point_augment_fn(points)
            if self.img_transform:
                image = self.img_transform(image)
            return points, label, image
        else:
            points, label = item
            if self.point_augment_fn:
                points = self.point_augment_fn(points)
            return points, label

    def __len__(self):
        return len(self.base_dataset)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed, device):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'cuda' in str(device):
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Stratified Split
# ---------------------------------------------------------------------------

def stratified_split(dataset, n_train_per_class=60, n_val_per_class=10,
                     n_test_per_class=10, seed=42):
    """Stratified train/val/test split for balanced datasets.

    Currently hardcoded to expect 80 samples per class (TU-Berlin).
    TODO: Must be rewritten for Part B (medical dataset with different class sizes).
    """
    np.random.seed(seed)

    total_per_class = n_train_per_class + n_val_per_class + n_test_per_class
    assert total_per_class == 80, f"Total per class must be 80, got {total_per_class}"

    # Collect indices for each class — handles both 2-tuple and 3-tuple datasets
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        item = dataset[idx]
        label = item[1]  # label is always the second element
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def predefined_split_dataset(dataset, train_file, val_file, test_file):
    """Split dataset using predefined file lists.

    Each file contains lines of format: '<moca_score> <collection>/<filename>.png'
    Matches entries to dataset samples by (subfolder, stem).

    Args:
        dataset: AlzheimerDataset instance.
        train_file: Path to text file listing training samples.
        val_file: Path to text file listing validation samples.
        test_file: Path to text file listing test samples.

    Returns:
        (train_subset, val_subset, test_subset) as Subsets.
    """
    # Build lookup: (subfolder, stem) -> dataset index
    sample_to_idx = {}
    for idx, (_, subfolder, stem) in enumerate(dataset.samples):
        sample_to_idx[(subfolder, stem)] = idx

    def _parse_split_file(filepath):
        indices = []
        missing = 0
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                rel_path = parts[1]  # e.g. 'India2/R_4o5NUStgBWnRJVY.png'
                collection, filename = rel_path.split('/', 1)
                stem = os.path.splitext(filename)[0]
                key = (collection, stem)
                if key in sample_to_idx:
                    indices.append(sample_to_idx[key])
                else:
                    missing += 1
        if missing:
            print(f"Warning: {missing} entries in {filepath} not found in dataset")
        return indices

    train_indices = _parse_split_file(train_file)
    val_indices = _parse_split_file(val_file)
    test_indices = _parse_split_file(test_file)

    print(f"Predefined split: {len(train_indices)} train, {len(val_indices)} val, "
          f"{len(test_indices)} test (total {len(train_indices) + len(val_indices) + len(test_indices)})")

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def random_split_dataset(dataset, train_frac=0.7, val_frac=0.15, seed=42):
    """Random train/val/test split by fraction.

    Args:
        dataset: Any PyTorch Dataset.
        train_frac: Fraction of data for training (default 0.7).
        val_frac: Fraction of data for validation (default 0.15).
            Test fraction is 1 - train_frac - val_frac.
        seed: Random seed.

    Returns:
        (train_subset, val_subset, test_subset) as Subsets.
    """
    np.random.seed(seed)
    n = len(dataset)
    indices = np.random.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:n_train + n_val].tolist()
    test_indices = indices[n_train + n_val:].tolist()

    print(f"Split: {len(train_indices)} train, {len(val_indices)} val, "
          f"{len(test_indices)} test (total {n})")

    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def regression_metrics(predictions, targets):
    """Compute MAE, RMSE, and R-squared for regression evaluation.

    Args:
        predictions: Tensor of predicted values.
        targets: Tensor of ground truth values.

    Returns:
        dict with 'mae', 'rmse', 'r2' keys.
    """
    predictions = predictions.float()
    targets = targets.float()

    ae = torch.abs(predictions - targets)
    se = (predictions - targets) ** 2

    mae = ae.mean().item()
    rmse = torch.sqrt(se.mean()).item()

    ss_res = se.sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"mae": mae, "rmse": rmse, "r2": r2}


def accuracy_topk(output, target, topk=(1, 5)):
    """Compute top-k accuracy for the specified values of k.

    Args:
        output: Model logits, shape (B, num_classes).
        target: Ground truth labels, shape (B,).
        topk: Tuple of k values to compute.

    Returns:
        List of tensors, one per k, each containing the accuracy percentage.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class PointCloudDataset(Dataset):
    def __init__(self, root_dir, num_points=1024, grid_size=4):
        self.root_dir = root_dir
        self.files = sorted(os.listdir(root_dir))
        self.num_points = num_points
        self.grid_size = grid_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        data = torch.load(file_path)
        point_cloud = data['point_cloud']
        label = data['label']
        return point_cloud, label

    def images_to_point_clouds(self, images_path):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            # transforms.Resize((1024, 1024)),
            CustomSketchToPointCloud(num_points=self.num_points, grid_size=self.grid_size)
        ])

        dataset = datasets.ImageFolder(root=images_path, transform=transform)
        save_dir = self.root_dir
        os.makedirs(save_dir, exist_ok=True)

        for idx, (image, label) in tqdm(enumerate(dataset), total=len(dataset)):
            point_cloud = image
            save_path = os.path.join(save_dir, f"{idx:05d}.pt")
            torch.save({'point_cloud': point_cloud, 'label': label}, save_path)

        self.files = sorted(os.listdir(self.root_dir))

    @staticmethod
    def save_point_cloud_plots(dataset, out_path):
        for i in range(len(dataset)):

            points, label = dataset[i]

            plt.scatter(points[:, 1], points[:, 0], color='red', s=0.1)
            plt.xlim(-1, 1)
            plt.ylim(1, -1)
            plt.show()

            filename = f"plot{i}.png"

            save_path = os.path.join(out_path, filename)

            plt.savefig(save_path)
            plt.close()


class CustomSketchToPointCloud:
    def __init__(self, num_points=1024, grid_size=4):
        self.num_points = num_points
        self.grid_size = grid_size

    def __call__(self, image):
        image = np.array(image)
        if np.max(image) > 1:
            image = image / 255
        num_of_all_blacks = (image < 0.98).sum()
        assert num_of_all_blacks > 0
        height, width = image.shape
        all_black_pixels = np.empty((0, 2))
        patches = CustomSketchToPointCloud.split_image_into_parts(image=image, n=self.grid_size)
        reminder = 1

        for patch, x, y in patches:
            num_blacks = (patch < 0.98).sum()
            num_blacks_to_pick = (num_blacks * self.num_points) / num_of_all_blacks
            if reminder >= 1:
                reminder -= math.ceil(num_blacks_to_pick) - num_blacks_to_pick
                num_blacks_to_pick = math.ceil(num_blacks_to_pick)
            else:
                reminder += num_blacks_to_pick - math.floor(num_blacks_to_pick)
                num_blacks_to_pick = math.floor(num_blacks_to_pick)

            black_pixels = np.argwhere(patch < 0.98)
            if black_pixels.shape[0] > 0:
                chosen_blacks_indices = np.random.choice(black_pixels.shape[0], size=num_blacks_to_pick, replace=False)
                chosen_blacks = black_pixels[chosen_blacks_indices]
                chosen_blacks += np.array([x, y])
                all_black_pixels = np.append(all_black_pixels, chosen_blacks, axis=0)
        assert all_black_pixels.shape[0] == self.num_points
        all_black_pixels[:, 0] = (all_black_pixels[:, 0] / (height - 1)) * 2 - 1  # y-coordinate
        all_black_pixels[:, 1] = (all_black_pixels[:, 1] / (width - 1)) * 2 - 1  # x-coordinate
        return torch.tensor(all_black_pixels, dtype=torch.float32)

    @staticmethod
    def split_image_into_parts(image, n):
        if isinstance(image, Image.Image):
            image = np.array(image)

        height, width = image.shape

        patches = []
        for i in range(0, height, n):
            for j in range(0, width, n):
                # Define the current patch's boundaries
                patch = [image[i:i + n, j:j + n], i, j]
                # If the patch exceeds the image boundaries, handle it
                if patch[0].shape[0] == n and patch[0].shape[1] == n:
                    patches.append(patch)
        return patches