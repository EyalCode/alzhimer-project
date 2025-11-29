import torch
from torch.nn import CrossEntropyLoss, NLLLoss
import matplotlib.pyplot as plt
from util import plot_point_cloud
from datasets import TUBerlinDataset
from model import PointNet2D, PointNetPlusPlus
from train import PointNet2dClassifierTrainer
from util import AugmentedDataset
from util import PointCloudAugmentation
from util import set_seed
from util import stratified_split
from pathlib import Path
import os

args = {
    'data_dir': 'preprocessed_point_clouds',
    'num_points': 1024,
    'num_classes': 250,
    'batch_size': 16,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'checkpoint_path': 'TUBerlin_checkpoints_16.pt'
}


def validate_model():
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

    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=TUBerlinDataset.collate_fn)

    # Initialize the model
    model = PointNetPlusPlus(num_class=args['num_classes'])
    loss_fn = NLLLoss()

    # Load model checkpoint
    Path(os.path.dirname(args['checkpoint_path'])).mkdir(exist_ok=True)
    if os.path.isfile(args['checkpoint_path']):
        print(f"*** Loading checkpoint file {args['checkpoint_path']}")
        saved_state = torch.load(args['checkpoint_path'], map_location=args['device'])
        model.load_state_dict(saved_state["model_state"])
    else:
        print(f"No checkpoint found at {args['checkpoint_path']}. Exiting validation.")
        return

    model.to(args['device'])

    # Evaluate the model
    print("Evaluating the model on the validation set...")
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in validation_loader:
            points, labels = batch
            points, labels = points.to(args['device']), labels.to(args['device'])
            
            outputs = model(points)
            if isinstance(outputs, tuple) or isinstance(outputs, list):
                outputs = outputs[0]
    
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(validation_loader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    validate_model()
