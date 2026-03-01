import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import random
import os
from collections import defaultdict
from torch.utils.data import Subset
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import Dataset


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
                need_replace = num_blacks_to_pick > black_pixels.shape[0]
                chosen_blacks_indices = np.random.choice(black_pixels.shape[0], size=num_blacks_to_pick, replace=need_replace)
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
