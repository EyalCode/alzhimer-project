import torch
from torch.nn import CrossEntropyLoss, NLLLoss
import matplotlib.pyplot as plt
from util import plot_point_cloud
from datasets import TUBerlinDataset
from model import PointNet2D, PointNetPlusPlus,PointNetResNetFusion,PointNetConvNextFusion, PointNetConvNextFusionBase
from train import PointNet2dClassifierTrainer
from util import AugmentedDataset
from util import PointCloudAugmentation
from util import set_seed
from util import stratified_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from collections import Counter, defaultdict
from torchvision import transforms
from util import accuracy_topk
import os
import random

args = {
    'point_clouds_data_dir': 'preprocessed_point_clouds',
    'images_data_dir': 'sketches_png/png',
    'num_points': 1024,
    'num_classes': 250,
    'batch_size': 64,
    'num_epochs': 0,
    'learning_rate_pointnet': 5e-5,
    'learning_rate_cnn': 5e-6,
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

TRAIN_IMG_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=90), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# B. Validation/Test Transform (Deterministic - NO Randomness)
VAL_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(256),      # Resize to slightly larger
    transforms.CenterCrop(224),  # Crop the exact center
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":

    g = torch.Generator()
    g.manual_seed(args['seed'])

    # Initialize Dataset with BOTH directories
    dataset = TUBerlinDataset(
        data_dir=args['point_clouds_data_dir'],
        images_dir=args['images_data_dir']
    )
    
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size 
  
    # Assuming stratified_split is imported or defined elsewhere
    train_dataset, validation_dataset, test_dataset = stratified_split(dataset=dataset, seed=args['seed'])

    augment = PointCloudAugmentation() # Assuming this is defined
    train_dataset = AugmentedDataset(train_dataset,img_transform=TRAIN_IMG_TRANSFORM,point_augment_fn=augment)
    validation_dataset = AugmentedDataset(validation_dataset, img_transform=VAL_IMG_TRANSFORM,point_augment_fn=None)
    test_dataset = AugmentedDataset(test_dataset, img_transform=VAL_IMG_TRANSFORM,point_augment_fn=None)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=TUBerlinDataset.collate_fn)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=TUBerlinDataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=TUBerlinDataset.collate_fn)


    print("Initializing Fusion Model...")
    model = PointNetConvNextFusionBase(num_class=args['num_classes']).to(args['device'])
    
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "cnn" in name:
            backbone_params.append(param)
        else: # PointNet, Fusion Layers, etc.
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args['learning_rate_cnn']}, # SLOW: Fine-tune backbone
        {'params': head_params,     'lr': args['learning_rate_pointnet']}  # FAST: Train PointNet/Head quickly
    ], weight_decay=0.1)
    

    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=7e-8)
    loss_fn = CrossEntropyLoss(label_smoothing=0.1)

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

    first_linear_layer = model.fusion_fc[0]
    fc1_weights = first_linear_layer.weight.data.abs()

    # Calculate average strength of connections
    point_strength = fc1_weights[:, :1024].mean().item()
    img_strength   = fc1_weights[:, 1024:].mean().item()
    
    print(f"PointNet Connection Strength: {point_strength:.5f}")
    print(f"ConvNext Connection Strength:   {img_strength:.5f}")
    
    ratio = img_strength / point_strength
    print(f"Ratio: ConvNext is {ratio:.2f}x stronger than PointNet")


    # Ensure this matches the folder name in your args
    data_dir = 'sketches_png/png'
    
    try:
        # Get list of all subfolders (class names)
        class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        # Sort them alphabetically (Standard dataset loaders assign ID 0 to the first alphabetical name)
        class_names.sort()
    
    except FileNotFoundError:
      print(f"Error: Could not find the folder '{data_dir}'. Make sure the path is correct.")
      
    ID_TO_NAME = { i: class_names[i] for i in range(len(class_names))}

    print("\n" + "="*50)
    print("STARTING POST-TRAINING EVALUATION")
    print("="*50)

    model.eval()  # Set model to evaluation mode

    # Accumulators for Weighted Average
    top1_sum = 0
    top2_sum = 0
    top3_sum = 0
    total_test = 0
    
    print("\nStarting Evaluation on Test Set...")
    
    with torch.no_grad():
        for points, labels, images in test_loader:

            # Move ALL data to the device
            points = points.to(args['device'])
            labels = labels.to(args['device'])
            images = images.to(args['device'])

            # Forward pass
            outputs = model(points, images)

            # Handle tuple output (logits, features)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take only the logits

            # --- CALCULATE TOP-1, TOP-2, TOP-3 ---
            # We pass topk=(1, 2, 3) to get all three metrics at once
            acc1, acc2, acc3 = accuracy_topk(outputs, labels, topk=(1, 2, 3))

            # accuracy_topk returns the batch average percentage (e.g., 90.0)
            # We multiply by batch size to get the "weighted sum" for the final average
            bs = labels.size(0)
            top1_sum += acc1.item() * bs
            top2_sum += acc2.item() * bs
            top3_sum += acc3.item() * bs
            total_test += bs

    # Calculate Final Averages
    final_top1 = top1_sum / total_test
    final_top2 = top2_sum / total_test
    final_top3 = top3_sum / total_test
    
    print(f"\nFinal Performance on Test Set:")
    print(f"------------------------------")
    print(f"Top-1 Accuracy: {final_top1:.2f}% (Exact Match)")
    print(f"Top-2 Accuracy: {final_top2:.2f}% (Correct class in top 2 guesses)")
    print(f"Top-3 Accuracy: {final_top3:.2f}% (Correct class in top 3 guesses)")
    print(f"------------------------------")

    # 2. Distribution of False Classifications (Validation Set)
    mistake_counts = torch.zeros(args['num_classes'], dtype=torch.long, device=args['device'])
    total_mistakes = 0

    # Stores count of what each class was confused with: {true_label: Counter({pred_label: count})}
    confusion_stats = defaultdict(Counter) 
    # Stores one point cloud example for every class that had a mistake: {true_label: point_cloud_data}
    mistake_examples = {} 
    
    print("\nAnalyzing Validation Set Errors...")
    
    with torch.no_grad():
        for points, labels, images in validation_loader:

            points = points.to(args['device'])
            labels = labels.to(args['device'])
            images = images.to(args['device'])

            outputs = model(points, images)
            
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Take only the logits
            
            _, predicted = torch.max(outputs.data, 1)
            
            # Identify which samples were wrong
            mistake_mask = (predicted != labels)
            
            if mistake_mask.sum() > 0:
                # Get the ground truth labels of the misclassified samples
                mistake_labels = labels[mistake_mask]
                
                # Count occurrences of each class within the mistakes
                batch_mistake_counts = torch.bincount(mistake_labels, minlength=args['num_classes'])
                mistake_counts += batch_mistake_counts
                total_mistakes += mistake_labels.size(0)

                # --- Collect specific details for the top 3 analysis ---
                # Get indices of mistakes in this batch
                mistake_indices = torch.where(mistake_mask)[0]
                
                for idx in mistake_indices:
                    true_lbl = labels[idx].item()
                    pred_lbl = predicted[idx].item()
                    
                    # Record what it was confused with
                    confusion_stats[true_lbl][pred_lbl] += 1
                    
                    # Save the POINT CLOUD for plotting later
                    # (We use 'points' here, not 'inputs' anymore)
                    if true_lbl not in mistake_examples:
                        mistake_examples[true_lbl] = points[idx].cpu()

    # Calculate and Print Distribution
    print(f"\nDistribution of False Classifications (Validation Set):")
    print(f"Total Validation Mistakes: {total_mistakes}")

    if total_mistakes > 0:
        results = []
        for class_id in range(args['num_classes']):
            count = mistake_counts[class_id].item()
            if count > 0:
                percentage = (count / total_mistakes) * 100
                results.append((class_id, count, percentage))
        
        # Sort by percentage (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Print table (Now using Class Names)
        print(f"{'Class Name':<25} | {'Mistake Count':<15} | {'% of All Mistakes':<20}")
        print("-" * 65)
        for class_id, count, percentage in results[:20]:
            name = ID_TO_NAME.get(class_id, f"ID {class_id}")
            print(f"{name:<25} | {count:<15} | {percentage:<20.2f}")
            
        if len(results) > 20:
            print(f"... and {len(results) - 20} other classes with errors.")

        # --- NEW: Top 3 Analysis (Specific Confusion + Images) ---
        print("\n" + "="*50)
        print("TOP 3 HARDEST CLASSES ANALYSIS")
        print("="*50)

        top_3_errors = results[:3]

        for i, (class_id, count, pct) in enumerate(top_3_errors):
            # TRANSLATION HAPPENS HERE
            class_name = ID_TO_NAME.get(class_id, f"ID {class_id}")

            print(f"\n{i+1}. Class: {class_name} (ID: {class_id})")
            print(f"   Total Errors: {count}")
            
            # Show what it was confused with
            print("   Most often confused with:")
            common_confusions = confusion_stats[class_id].most_common(5) 
            
            for pred_id, freq in common_confusions:
                pred_name = ID_TO_NAME.get(pred_id, f"ID {pred_id}")
                print(f"    -> {pred_name} (ID: {pred_id}): {freq} times")

            if class_id in mistake_examples:
                top_confused_id = common_confusions[0][0]
                pred_name_top = ID_TO_NAME.get(top_confused_id, f"ID {top_confused_id}")

                print(f"   Displaying example of '{class_name}' misclassified as '{pred_name_top}'...")
                
                # Retrieve point cloud
                pc_data = mistake_examples[class_id]
                
                if isinstance(pc_data, torch.Tensor):
                    pc_data = pc_data.numpy()
                
                plt.figure(figsize=(8, 8))
                plt.title(f"True: {class_name} | Predicted: {pred_name_top}")
                
                plot_point_cloud(pc_data)
                plt.show()


    print("\n" + "=" * 50)
    print("VISUALIZING AUGMENTATION EXAMPLES (TRAIN SET)")
    print("=" * 50)

    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        t = tensor.clone().detach().cpu()
        t = t * std + mean
        return t.clamp(0, 1).permute(1, 2, 0)


    # Pick 5 random indices from the training set
    indices = random.sample(range(len(train_dataset)), 5)

    for i, idx in enumerate(indices):
        # train_dataset.base_dataset accesses the data BEFORE augmentation
        orig_points, label_idx, orig_img_pil = train_dataset.base_dataset[idx]

        # Accessing train_dataset[idx] triggers the augmentation logic
        aug_points, _, aug_img_tensor = train_dataset[idx]

        class_name = ID_TO_NAME.get(label_idx, f"ID {label_idx}")

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"Example {i + 1}: Class '{class_name}'", fontsize=16)

        # A: Original Points
        op = orig_points.numpy() if isinstance(orig_points, torch.Tensor) else orig_points
        if op.shape[0] == 2: op = op.T
        axs[0].scatter(op[:, 0], op[:, 1], s=5, c='blue', alpha=0.5)
        axs[0].set_title("Original Point Cloud")
        axs[0].axis('equal')

        # B: Augmented Points
        ap = aug_points.numpy() if isinstance(aug_points, torch.Tensor) else aug_points
        if ap.shape[0] == 2: ap = ap.T
        axs[1].scatter(ap[:, 0], ap[:, 1], s=5, c='blue', alpha=0.5)
        axs[1].set_title("Augmented Point Cloud")
        axs[1].axis('equal')

        # C: Original Image
        axs[2].imshow(orig_img_pil)
        axs[2].set_title("Original Image")
        axs[2].axis('off')

        # D: Augmented Image
        axs[3].imshow(denormalize(aug_img_tensor))
        axs[3].set_title("Augmented Image")
        axs[3].axis('off')

        plt.tight_layout()
        plt.show()