import torch
from torch.utils.data import Dataset
import numpy as np
import os

TUBERLIN_DATASET_SIZE = 20000

class TUBerlinDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.size = TUBERLIN_DATASET_SIZE

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Check if the index is valid
        if idx < 0 or idx >= self.size:
            raise IndexError("Index out of range")
        
        # Load the point cloud data from the file
        file_path = os.path.join(self.data_dir, f'{idx:05d}.pt')
        item_dict = torch.load(file_path)
        point_cloud = item_dict['point_cloud'].transpose(0, 1)  # Transpose to (N, C) format
        label = item_dict['label']
        return point_cloud, label
    
    @staticmethod
    def collate_fn(batch):
        point_clouds, labels = zip(*batch)
        point_clouds = torch.stack(point_clouds, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return point_clouds, labels
    
