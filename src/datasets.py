"""
Dataset loaders for point cloud and image data.

TUBerlinDataset loads pre-processed point clouds (.pt files) and optionally
the corresponding sketch images (.png files) for fusion models.

AlzheimerDataset loads point clouds with MOCA score labels from the
Alzheimer's self-portrait dataset.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class TUBerlinDataset(Dataset):
    """TU-Berlin sketch dataset with point clouds and optional images.

    When images_dir is None, returns (point_cloud, label).
    When images_dir is provided, returns (point_cloud, label, image).

    Point clouds are stored as .pt files with keys 'point_cloud' (C, N) and 'label'.
    Images are loaded as PIL RGB from sorted class subfolders.

    Args:
        data_dir: Path to directory containing .pt point cloud files (e.g. 00000.pt, 00001.pt, ...)
        images_dir: Optional path to directory containing class subfolders with .png images.
    """

    def __init__(self, data_dir, images_dir=None):
        self.data_dir = data_dir
        self.images_dir = images_dir

        # Dynamically detect dataset size from .pt files
        self.pt_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        self.size = len(self.pt_files)

        # Load image paths if images_dir is provided
        self.image_paths = []
        if self.images_dir is not None:
            self._load_image_paths()
            if len(self.image_paths) != self.size:
                print(f"Warning: Found {len(self.image_paths)} images but {self.size} point clouds.")

    def _load_image_paths(self):
        """Scan class subfolders and collect .png paths in sorted order
        to align with point cloud indices (00000, 00001, ...).
        """
        classes = sorted([
            d for d in os.listdir(self.images_dir)
            if os.path.isdir(os.path.join(self.images_dir, d))
        ])

        for class_name in classes:
            class_folder = os.path.join(self.images_dir, class_name)
            images = sorted(
                [f for f in os.listdir(class_folder) if f.endswith('.png')],
                key=lambda x: int(os.path.splitext(x)[0]) if x[:-4].isdigit() else x,
            )
            for img_name in images:
                self.image_paths.append(os.path.join(class_folder, img_name))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.size:
            raise IndexError("Index out of range")

        file_path = os.path.join(self.data_dir, f'{idx:05d}.pt')
        item_dict = torch.load(file_path, weights_only=True)
        point_cloud = item_dict['point_cloud'].transpose(0, 1)  # (C, N) → (N, C)
        label = item_dict['label']

        if self.images_dir is not None:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            return point_cloud, label, image

        return point_cloud, label

    @staticmethod
    def collate_fn(batch):
        """Stack batch items into tensors. Handles both 2-tuple and 3-tuple batches."""
        if len(batch[0]) == 3:
            point_clouds, labels, images = zip(*batch)
            point_clouds = torch.stack(point_clouds, dim=0)
            labels = torch.tensor(labels, dtype=torch.long)
            images = torch.stack(images, dim=0)
            return point_clouds, labels, images
        else:
            point_clouds, labels = zip(*batch)
            point_clouds = torch.stack(point_clouds, dim=0)
            labels = torch.tensor(labels, dtype=torch.long)
            return point_clouds, labels


class AlzheimerDataset(Dataset):
    """Alzheimer's self-portrait dataset with point clouds and MOCA scores.

    Loads .pt files from a directory with subfolders (e.g. Aviel/, India/, ...).
    Each .pt file contains {'point_cloud': (N, 2), 'label': int}.
    Returns point clouds in channels-first format (2, N) for model compatibility.

    When images_dir is provided, also loads the corresponding source image.

    Supports two task modes:
        - "regression": labels are float MOCA scores (0–30)
        - "classification": labels are int class indices mapped from MOCA scores:
            0 = Demented (MoCA 0–19), 1 = Mild (MoCA 20–25), 2 = High Cognitive (MoCA 26–30)

    Args:
        data_dir: Path to root of point cloud subfolders (e.g. 'alzhimer_point_clouds/').
        images_dir: Optional path to source images with matching subfolder/filename structure.
        task: "regression" (default) or "classification".
    """

    IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

    MOCA_LEVELS = {
        "Demented": (0, 19),
        "Mild": (20, 25),
        "High Cognitive": (26, 30),
    }

    @staticmethod
    def moca_to_class(score):
        """Map a MoCA score (0–30) to a class index.

        Returns: 0 (Demented, 0–19), 1 (Mild, 20–25), 2 (High Cognitive, 26–30).
        """
        if score <= 19:
            return 0
        elif score <= 25:
            return 1
        else:
            return 2

    def __init__(self, data_dir, images_dir=None, moca_translation=False, task="regression"):
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.moca_translation = moca_translation
        self.task = task
        self.samples = []  # list of (pt_path, subfolder, stem)

        for subfolder in sorted(os.listdir(data_dir)):
            subfolder_path = os.path.join(data_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            for fname in sorted(os.listdir(subfolder_path)):
                if fname.endswith('.pt'):
                    stem = os.path.splitext(fname)[0]
                    pt_path = os.path.join(subfolder_path, fname)
                    self.samples.append((pt_path, subfolder, stem))

        # Pre-resolve image paths if images_dir is provided
        self.image_paths = {}
        if self.images_dir is not None:
            self._resolve_image_paths()

    def _resolve_image_paths(self):
        """Match each sample to its source image by subfolder/stem."""
        for idx, (_, subfolder, stem) in enumerate(self.samples):
            img_folder = os.path.join(self.images_dir, subfolder)
            if not os.path.isdir(img_folder):
                continue
            for ext in self.IMG_EXTS:
                img_path = os.path.join(img_folder, stem + ext)
                if os.path.isfile(img_path):
                    self.image_paths[idx] = img_path
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pt_path, _, _ = self.samples[idx]
        item_dict = torch.load(pt_path, weights_only=True)
        point_cloud = item_dict['point_cloud'].T  # (N, 2) → (2, N) channels-first
        label = float(item_dict['label'])
        if self.moca_translation and label <= 12.0:
            label = label / 3.0 + 12.0

        if self.task == "classification":
            label = self.moca_to_class(label)

        if self.images_dir is not None:
            img_path = self.image_paths.get(idx)
            if img_path is not None:
                image = Image.open(img_path).convert('RGB')
            else:
                raise FileNotFoundError(f"No source image found for {pt_path}")
            return point_cloud, label, image

        return point_cloud, label

    @staticmethod
    def make_collate_fn(task="regression"):
        """Return a collate function with the appropriate label dtype.

        For regression: labels are torch.float32 (MOCA scores).
        For classification: labels are torch.long (class indices).
        """
        label_dtype = torch.long if task == "classification" else torch.float32

        def collate_fn(batch):
            if len(batch[0]) == 3:
                point_clouds, labels, images = zip(*batch)
                point_clouds = torch.stack(point_clouds, dim=0)
                labels = torch.tensor(labels, dtype=label_dtype)
                images = torch.stack(images, dim=0)
                return point_clouds, labels, images
            else:
                point_clouds, labels = zip(*batch)
                point_clouds = torch.stack(point_clouds, dim=0)
                labels = torch.tensor(labels, dtype=label_dtype)
                return point_clouds, labels

        return collate_fn

    @staticmethod
    def collate_fn(batch):
        """Stack batch items into tensors. Labels are float (MOCA scores).

        Legacy static method for backward compatibility. Prefer make_collate_fn().
        """
        if len(batch[0]) == 3:
            point_clouds, labels, images = zip(*batch)
            point_clouds = torch.stack(point_clouds, dim=0)
            labels = torch.tensor(labels, dtype=torch.float32)
            images = torch.stack(images, dim=0)
            return point_clouds, labels, images
        else:
            point_clouds, labels = zip(*batch)
            point_clouds = torch.stack(point_clouds, dim=0)
            labels = torch.tensor(labels, dtype=torch.float32)
            return point_clouds, labels
