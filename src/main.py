import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from util import plot_point_cloud
from datasets import TUBerlinDataset
from model import PointNet2D
from train import PointNet2dClassifierTrainer

args = {
    'data_dir': '../preprocessed_point_clouds',
    'num_points': 1024,
    'num_classes': 250,
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 0.0001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

if __name__ == "__main__":
    # Load the dataset
    data_dir = args['data_dir']
    dataset = TUBerlinDataset(data_dir)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=TUBerlinDataset.collate_fn)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=False, collate_fn=TUBerlinDataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=TUBerlinDataset.collate_fn)


    # Initialize the model
    model = PointNet2D(num_classes=args['num_classes'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_fn = CrossEntropyLoss()

    # Train the model
    trainer = PointNet2dClassifierTrainer(
        model,
        loss_fn,
        optimizer,
        scheduler,
        args['device']
    )

    fit_result = trainer.fit(
        dl_train = train_loader,
        dl_test = validation_loader,
        num_epochs = args['num_epochs'],
        checkpoints = "TUBerlin_checkpoints",
        early_stopping= 30
    )
