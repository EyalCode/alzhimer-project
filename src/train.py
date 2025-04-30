import torch
import torch.nn as nn
import torch.nn.functional as F
from util import plot_point_cloud


def train(model, train_loader, optimizer, epoch, device='cuda'):
    model.train()   
    running_loss = 0.0

    for batch_idx, (point_cloud, label) in enumerate(train_loader):
        point_cloud = point_cloud.to(device)
        label = label.to(device)
                
        optimizer.zero_grad()
        output = model(point_cloud)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(point_cloud)}/{len(train_loader.dataset)} ' + \
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch} complete. Average loss: {avg_loss:.6f}')
    return avg_loss

