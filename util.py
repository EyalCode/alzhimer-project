import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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