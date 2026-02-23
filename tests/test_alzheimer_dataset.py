"""
Tests for AlzheimerDataset in src/datasets.py.

Run from project root:
    python -m pytest tests/test_alzheimer_dataset.py -v
"""

import sys
import os
import unittest
import random

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from datasets import AlzheimerDataset

DATA_DIR    = 'alzhimer_point_clouds'
IMAGES_DIR  = 'alzhimer_src_images'
N_POINTS    = 1024
BATCH_SIZE  = 8


class TestAlzheimerDatasetPointsOnly(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ds = AlzheimerDataset(DATA_DIR)

    # ------------------------------------------------------------------ size
    def test_dataset_size(self):
        self.assertEqual(len(self.ds), 1595)

    # ---------------------------------------------------------------- shapes
    def test_point_cloud_shape(self):
        pc, _ = self.ds[0]
        self.assertEqual(pc.shape, torch.Size([2, N_POINTS]),
                         "Point cloud must be (2, N) channels-first")

    def test_point_cloud_dtype(self):
        pc, _ = self.ds[0]
        self.assertEqual(pc.dtype, torch.float32)

    # ---------------------------------------------------------------- labels
    def test_label_is_float(self):
        _, label = self.ds[0]
        self.assertIsInstance(label, float)

    def test_label_moca_range(self):
        """Spot-check 50 random samples — all MOCA scores must be in [0, 30]."""
        indices = random.sample(range(len(self.ds)), 50)
        for idx in indices:
            _, label = self.ds[idx]
            self.assertGreaterEqual(label, 0.0,
                                    f"MOCA score {label} below 0 at idx {idx}")
            self.assertLessEqual(label, 30.0,
                                 f"MOCA score {label} above 30 at idx {idx}")

    # ---------------------------------------------------------------- return type
    def test_returns_2tuple_without_images(self):
        item = self.ds[0]
        self.assertEqual(len(item), 2)

    # ---------------------------------------------------------------- boundary
    def test_first_and_last_accessible(self):
        self.ds[0]
        self.ds[len(self.ds) - 1]

    # ---------------------------------------------------------------- collate
    def test_collate_fn_shapes(self):
        batch = [self.ds[i] for i in range(BATCH_SIZE)]
        pcs, labels = AlzheimerDataset.collate_fn(batch)
        self.assertEqual(pcs.shape,    torch.Size([BATCH_SIZE, 2, N_POINTS]))
        self.assertEqual(labels.shape, torch.Size([BATCH_SIZE]))

    def test_collate_fn_label_dtype(self):
        batch = [self.ds[i] for i in range(BATCH_SIZE)]
        _, labels = AlzheimerDataset.collate_fn(batch)
        self.assertEqual(labels.dtype, torch.float32,
                         "Labels must be float32 for regression")

    # ---------------------------------------------------------------- dataloader
    def test_dataloader_batch(self):
        dl = DataLoader(self.ds, batch_size=BATCH_SIZE,
                        collate_fn=AlzheimerDataset.collate_fn, shuffle=False)
        pcs, labels = next(iter(dl))
        self.assertEqual(pcs.shape,    torch.Size([BATCH_SIZE, 2, N_POINTS]))
        self.assertEqual(labels.dtype, torch.float32)


class TestAlzheimerDatasetWithImages(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ds = AlzheimerDataset(DATA_DIR, images_dir=IMAGES_DIR)

    def test_returns_3tuple_with_images(self):
        item = self.ds[0]
        self.assertEqual(len(item), 3)

    def test_image_is_pil(self):
        from PIL import Image
        _, _, img = self.ds[0]
        self.assertIsInstance(img, Image.Image)

    def test_image_is_rgb(self):
        _, _, img = self.ds[0]
        self.assertEqual(img.mode, 'RGB')

    def test_collate_fn_3tuple_shapes(self):
        from torchvision import transforms
        to_tensor = transforms.ToTensor()
        batch = []
        for i in range(BATCH_SIZE):
            pc, label, img = self.ds[i]
            batch.append((pc, label, to_tensor(img)))
        pcs, labels, imgs = AlzheimerDataset.collate_fn(batch)
        self.assertEqual(pcs.shape,    torch.Size([BATCH_SIZE, 2, N_POINTS]))
        self.assertEqual(labels.shape, torch.Size([BATCH_SIZE]))
        self.assertEqual(imgs.shape[0], BATCH_SIZE)
        self.assertEqual(imgs.shape[1], 3,  "Images must have 3 channels (RGB)")


if __name__ == '__main__':
    unittest.main(verbosity=2)
