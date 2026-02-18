import os
import torch
import torch.utils.data
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, models



TUBERLIN_DATASET_SIZE = 20000

class TUBerlinDataset(Dataset):
    def __init__(self, data_dir, images_dir):
        self.data_dir = data_dir
        self.images_dir = images_dir
        self.size = TUBERLIN_DATASET_SIZE

        self.image_paths = []
        self._load_image_paths()

        if len(self.image_paths) != self.size:
            print(f"Warning: Found {len(self.image_paths)} images but expected {self.size} point clouds.")

    def _load_image_paths(self):
        """
        Scans the subfolders (classes) and collects all .png paths 
        in a sorted order to align with point cloud indices (00000, 00001...)
        """

        classes = sorted([d for d in os.listdir(self.images_dir) 
                          if os.path.isdir(os.path.join(self.images_dir, d))])
        
        for class_name in classes:
            class_folder = os.path.join(self.images_dir, class_name)
            

            images = [f for f in os.listdir(class_folder) if f.endswith('.png')]

            images.sort(key=lambda x: int(os.path.splitext(x)[0]) if x[:-4].isdigit() else x)
            
            for img_name in images:
                self.image_paths.append(os.path.join(class_folder, img_name))

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.size:
            raise IndexError("Index out of range")

        file_path = os.path.join(self.data_dir, f'{idx:05d}.pt')
        try:
            item_dict = torch.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            raise

        point_cloud = item_dict['point_cloud'].transpose(0, 1)
        label = item_dict['label']

        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise

        return point_cloud, label, image
    
    @staticmethod
    def collate_fn(batch):
        point_clouds, labels, images = zip(*batch)
        point_clouds = torch.stack(point_clouds, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        images = torch.stack(images, dim=0)
        return point_clouds, labels, images