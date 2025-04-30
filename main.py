import torch
import matplotlib.pyplot as plt
from util import plot_point_cloud
from datasets import TUBerlinDataset
from model import PointNet2D
from train import train

args = {
    'data_dir': 'preprocessed_point_clouds',
    'num_points': 1024,
    'num_classes': 250,
    'batch_size': 32,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

if __name__ == "__main__":
    # Load the dataset
    data_dir = args['data_dir']
    dataset = TUBerlinDataset(data_dir)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=TUBerlinDataset.collate_fn)

    # Initialize the model
    model = PointNet2D(num_classes=args['num_classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, epoch, device=args['device'])