import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from util import plot_point_cloud
from datasets import TUBerlinDataset
from model import PointNet2D, PointNetPlusPlus
from train import PointNet2dClassifierTrainer
from util import AugmentedDataset
from util import PointCloudAugmentation
from util import set_seed
from util import stratified_split

args = {
    'data_dir': '../preprocessed_point_clouds',
    'num_points': 1024,
    'num_classes': 250,
    'batch_size': 64,
    'num_epochs': 100,
    'learning_rate': 10e-4,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

if __name__ == "__main__":
    # Load the dataset
    set_seed(seed=args['seed'], device=args['device'])
    g = torch.Generator()
    g.manual_seed(args['seed'])

    data_dir = args['data_dir']
    dataset = TUBerlinDataset(data_dir)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size  # ensures sum equals dataset_size
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_dataset, validation_dataset, test_dataset = stratified_split(dataset=dataset, seed=args['seed'])

    augment = PointCloudAugmentation()
    train_dataset = AugmentedDataset(train_dataset, augment)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=TUBerlinDataset.collate_fn)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=TUBerlinDataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=TUBerlinDataset.collate_fn)


    # Initialize the model
    model = PointNetPlusPlus(num_class=args['num_classes'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['learning_rate'], betas=(0.9, 0.999), weight_decay=12e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)
    loss_fn = CrossEntropyLoss(label_smoothing=0.10)

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
        early_stopping= 45
    )
