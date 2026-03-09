"""
Training and evaluation entry point.

Usage:
    python src/main.py --config config.json

All training arguments are read from the JSON config file.
Set training.num_epochs to 0 to skip training and run evaluation only.
"""

import argparse
import json
import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torchvision import transforms

from datasets import TUBerlinDataset, AlzheimerDataset
from model import (
    PointNet2D,
    PointNetPlusPlus,
    PointNetResNetFusion,
    PointNetConvNextFusion,
    PointNetConvNextFusionBase,
    load_pretrained_encoders,
)
from train import PointNet2dClassifierTrainer, RegressionTrainer
from util import (
    AugmentedDataset,
    PointCloudAugmentation,
    accuracy_topk,
    predefined_split_dataset,
    random_split_dataset,
    regression_metrics,
    set_seed,
    stratified_split,
)

# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "PointNet2D": {
        "class": PointNet2D,
        "needs_images": False,
    },
    "PointNetPlusPlus": {
        "class": PointNetPlusPlus,
        "needs_images": False,
    },
    "PointNetResNetFusion": {
        "class": PointNetResNetFusion,
        "needs_images": True,
    },
    "PointNetConvNextFusion": {
        "class": PointNetConvNextFusion,
        "needs_images": True,
    },
    "PointNetConvNextFusionBase": {
        "class": PointNetConvNextFusionBase,
        "needs_images": True,
    },
}

# ---------------------------------------------------------------------------
# Image Transforms (ImageNet normalization)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_IMG_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

VAL_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

def load_config(path):
    """Load and validate training configuration from a JSON file."""
    with open(path, 'r') as f:
        config = json.load(f)

    model_name = config.get("model")
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    return config


# ---------------------------------------------------------------------------
# Build Functions
# ---------------------------------------------------------------------------

def build_model(config, device):
    """Instantiate the selected model and move it to device.

    Returns:
        (model, model_info) where model_info is the registry entry dict.
    """
    model_name = config["model"]
    model_info = MODEL_REGISTRY[model_name]
    model_cls = model_info["class"]

    kwargs = {
        "num_classes": config.get("num_classes", 250),
    }
    # PointNet2D does not accept normal_channel
    if model_name != "PointNet2D":
        kwargs["normal_channel"] = config.get("use_normals", False)

    print(f"Initializing model: {model_name}")
    model = model_cls(**kwargs).to(device)
    return model, model_info


def build_dataloaders(config, needs_images):
    """Build train/val/test dataloaders from config.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_cfg = config["data"]
    training_cfg = config.get("training", {})
    aug_cfg = config.get("augmentation", {})
    batch_size = training_cfg.get("batch_size", 64)
    seed = config.get("seed", 42)

    # Build dataset
    images_dir = data_cfg.get("images_dir") if needs_images else None
    dataset = TUBerlinDataset(
        data_dir=data_cfg["point_clouds_dir"],
        images_dir=images_dir,
    )

    # Stratified split
    train_dataset, val_dataset, test_dataset = stratified_split(dataset, seed=seed)

    # Point cloud augmentation (disabled when augmentation config is false)
    use_augmentation = aug_cfg is not False and aug_cfg.get("enabled", True) is not False
    if use_augmentation:
        augment = PointCloudAugmentation(
            rotation_range=aug_cfg.get("rotation_range", 90),
            scale_range=aug_cfg.get("scale_range", (0.5, 1.5)),
            translation_range=aug_cfg.get("translation_range", 0.2),
            jitter_std=aug_cfg.get("jitter_std", 0.008),
            point_dropout_rate=aug_cfg.get("point_dropout_rate", 0.2),
        )
        train_img_transform = TRAIN_IMG_TRANSFORM
    else:
        augment = None
        train_img_transform = VAL_IMG_TRANSFORM
        print("Augmentation disabled — using val transforms for training")

    # Wrap datasets with augmentation
    if needs_images:
        train_dataset = AugmentedDataset(train_dataset, img_transform=train_img_transform, point_augment_fn=augment)
        val_dataset = AugmentedDataset(val_dataset, img_transform=VAL_IMG_TRANSFORM, point_augment_fn=None)
        test_dataset = AugmentedDataset(test_dataset, img_transform=VAL_IMG_TRANSFORM, point_augment_fn=None)
    else:
        train_dataset = AugmentedDataset(train_dataset, point_augment_fn=augment)
        val_dataset = AugmentedDataset(val_dataset, point_augment_fn=None)
        test_dataset = AugmentedDataset(test_dataset, point_augment_fn=None)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=TUBerlinDataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=TUBerlinDataset.collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=TUBerlinDataset.collate_fn,
    )

    return train_loader, val_loader, test_loader


def build_dataloaders_alzheimer(config, needs_images):
    """Build train/val/test dataloaders for the Alzheimer's dataset.

    Uses random split (configurable fractions) and AlzheimerDataset.collate_fn.
    """
    data_cfg = config["data"]
    training_cfg = config.get("training", {})
    aug_cfg = config.get("augmentation", {})
    batch_size = training_cfg.get("batch_size", 16)
    seed = config.get("seed", 42)

    images_dir = data_cfg.get("images_dir") if needs_images else None
    moca_translation = config.get("moca_translation", False)
    dataset = AlzheimerDataset(
        data_dir=data_cfg["point_clouds_dir"],
        images_dir=images_dir,
        moca_translation=moca_translation,
    )

    split_files = data_cfg.get("split_files")
    if split_files:
        train_dataset, val_dataset, test_dataset = predefined_split_dataset(
            dataset,
            train_file=split_files["train"],
            val_file=split_files["val"],
            test_file=split_files["test"],
        )
    else:
        train_frac = data_cfg.get("train_fraction", 0.7)
        val_frac = data_cfg.get("val_fraction", 0.15)
        train_dataset, val_dataset, test_dataset = random_split_dataset(
            dataset, train_frac, val_frac, seed,
        )

    use_augmentation = aug_cfg is not False and aug_cfg.get("enabled", True) is not False
    if use_augmentation:
        augment = PointCloudAugmentation(
            rotation_range=aug_cfg.get("rotation_range", 50),
            scale_range=aug_cfg.get("scale_range", (0.8, 1.2)),
            translation_range=aug_cfg.get("translation_range", 0.15),
            jitter_std=aug_cfg.get("jitter_std", 0.003),
            point_dropout_rate=aug_cfg.get("point_dropout_rate", 0.1),
        )
        train_img_transform = TRAIN_IMG_TRANSFORM
    else:
        augment = None
        train_img_transform = VAL_IMG_TRANSFORM
        print("Augmentation disabled — using val transforms for training")

    if needs_images:
        train_dataset = AugmentedDataset(train_dataset, img_transform=train_img_transform, point_augment_fn=augment)
        val_dataset = AugmentedDataset(val_dataset, img_transform=VAL_IMG_TRANSFORM, point_augment_fn=None)
        test_dataset = AugmentedDataset(test_dataset, img_transform=VAL_IMG_TRANSFORM, point_augment_fn=None)
    else:
        train_dataset = AugmentedDataset(train_dataset, point_augment_fn=augment)
        val_dataset = AugmentedDataset(val_dataset, point_augment_fn=None)
        test_dataset = AugmentedDataset(test_dataset, point_augment_fn=None)

    collate = AlzheimerDataset.collate_fn
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate,
    )

    return train_loader, val_loader, test_loader


def build_optimizer(model, config, needs_images):
    """Build optimizer with differential learning rates for fusion models."""
    training_cfg = config.get("training", {})
    lr = training_cfg.get("learning_rate", 5e-5)
    lr_cnn = training_cfg.get("learning_rate_cnn", 5e-6)
    weight_decay = training_cfg.get("weight_decay", 0.1)

    if needs_images:
        # Differential learning rates: CNN backbone gets slower LR
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "cnn" in name or "resnet" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": lr_cnn})
        if head_params:
            param_groups.append({"params": head_params, "lr": lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        print(f"Optimizer: AdamW with differential LRs (CNN: {lr_cnn}, rest: {lr})")
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=lr, weight_decay=weight_decay,
        )
        print(f"Optimizer: AdamW (lr: {lr})")

    return optimizer


def build_scheduler(optimizer, config):
    """Build learning rate scheduler from config."""
    training_cfg = config.get("training", {})
    scheduler_type = training_cfg.get("scheduler", "CosineAnnealingLR")

    if scheduler_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=training_cfg.get("scheduler_T_max", 50),
            eta_min=training_cfg.get("scheduler_eta_min", 7e-8),
        )
    elif scheduler_type == "StepLR":
        scheduler = StepLR(
            optimizer,
            step_size=training_cfg.get("scheduler_step_size", 30),
            gamma=training_cfg.get("scheduler_gamma", 0.9),
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    print(f"Scheduler: {scheduler_type}")
    return scheduler


def build_loss_fn(config):
    """Build loss function from config.

    For classification: CrossEntropyLoss with label smoothing.
    For regression: SmoothL1Loss or MSELoss.
    """
    task = config.get("task", "classification")
    training_cfg = config.get("training", {})

    if task == "regression":
        loss_type = training_cfg.get("loss", "SmoothL1")
        if loss_type == "SmoothL1":
            return torch.nn.SmoothL1Loss()
        elif loss_type == "MSE":
            return torch.nn.MSELoss()
        elif loss_type == "MAE":
            return torch.nn.L1Loss()
        else:
            raise ValueError(f"Unknown regression loss: {loss_type}")
    else:
        label_smoothing = training_cfg.get("label_smoothing", 0.1)
        return CrossEntropyLoss(label_smoothing=label_smoothing)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_topk(model, loader, device, needs_images):
    """Evaluate model on a dataloader and return Top-1/2/3 accuracy.

    Returns:
        (top1, top2, top3) as percentage floats.
    """
    model.eval()

    top1_sum = 0
    top2_sum = 0
    top3_sum = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            if needs_images:
                points, labels, images = batch
                points = points.to(device)
                labels = labels.to(device)
                images = images.to(device)
                outputs = model(points, images)
            else:
                points, labels = batch
                points = points.to(device)
                labels = labels.to(device)
                outputs = model(points)

            acc1, acc2, acc3 = accuracy_topk(outputs, labels, topk=(1, 2, 3))

            bs = labels.size(0)
            top1_sum += acc1.item() * bs
            top2_sum += acc2.item() * bs
            top3_sum += acc3.item() * bs
            total += bs

    top1 = top1_sum / total
    top2 = top2_sum / total
    top3 = top3_sum / total

    return top1, top2, top3


def evaluate_regression(model, loader, device, needs_images):
    """Evaluate regression model and return MAE, RMSE, R² metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            if needs_images:
                points, labels, images = batch
                points, labels, images = points.to(device), labels.to(device), images.to(device)
                outputs = model(points, images).squeeze(-1)
            else:
                points, labels = batch
                points, labels = points.to(device), labels.to(device)
                outputs = model(points).squeeze(-1)

            all_preds.append(outputs.cpu())
            all_targets.append(labels.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return regression_metrics(preds, targets)


# ---------------------------------------------------------------------------
# Encoder Freezing
# ---------------------------------------------------------------------------

# Encoder parameter prefixes per model — everything matching these is frozen.
ENCODER_PREFIXES = {
    "PointNet2D": ["conv1", "conv2", "conv3", "conv4", "conv5",
                   "bn1", "bn2", "bn3", "bn4", "bn5",
                   "feature_transform"],
    "PointNetPlusPlus": ["sa1", "sa2", "sa3"],
    "PointNetResNetFusion": ["sa1", "sa2", "sa3", "cnn"],
    "PointNetConvNextFusion": ["sa1", "sa2", "sa3", "cnn"],
    "PointNetConvNextFusionBase": ["sa1", "sa2", "sa3", "cnn"],
}


def freeze_encoders(model, model_name):
    """Freeze encoder parameters, leaving head layers trainable."""
    prefixes = ENCODER_PREFIXES.get(model_name, [])
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad = False
            frozen_count += 1
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Froze {frozen_count} encoder params, {trainable} params remain trainable")


def unfreeze_encoders(model, model_name, optimizer, lr):
    """Unfreeze encoder parameters and add them to the optimizer.

    Args:
        model: The model with frozen encoder params.
        model_name: Key into ENCODER_PREFIXES.
        optimizer: The existing optimizer — encoder params are added as a new group.
        lr: Learning rate for the newly unfrozen encoder params.
    """
    prefixes = ENCODER_PREFIXES.get(model_name, [])
    unfrozen = []
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            param.requires_grad = True
            unfrozen.append(param)

    if unfrozen:
        optimizer.add_param_group({"params": unfrozen, "lr": lr})

    print(f"Unfroze {len(unfrozen)} encoder params (lr={lr})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config):
    """Main training and evaluation loop.

    Supports both classification (TU-Berlin) and regression (Alzheimer's) tasks,
    dispatched via config["task"].
    """
    seed = config.get("seed", 42)
    task = config.get("task", "classification")
    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Task: {task}")

    set_seed(seed, device)

    # Build model
    model, model_info = build_model(config, device)
    needs_images = model_info["needs_images"]
    model_name = config["model"]

    # Transfer learning: load pretrained encoder weights
    pretrain_ckpt = config.get("pretrained_checkpoint")
    if pretrain_ckpt:
        print(f"\nLoading pretrained encoders from: {pretrain_ckpt}")
        loaded, skipped = load_pretrained_encoders(model, pretrain_ckpt, device)
        print(f"  Loaded {len(loaded)} params, skipped {len(skipped)}: {skipped}")

    # Freeze encoders if requested
    if config.get("freeze_encoders", False):
        freeze_encoders(model, model_name)

    # Build dataloaders
    if task == "regression":
        train_loader, val_loader, test_loader = build_dataloaders_alzheimer(config, needs_images)
    else:
        train_loader, val_loader, test_loader = build_dataloaders(config, needs_images)

    optimizer = build_optimizer(model, config, needs_images)
    scheduler = build_scheduler(optimizer, config)
    loss_fn = build_loss_fn(config)

    # Build trainer
    if task == "regression":
        trainer = RegressionTrainer(
            model, loss_fn, optimizer, scheduler, device, needs_images=needs_images,
        )
    else:
        trainer = PointNet2dClassifierTrainer(
            model, loss_fn, optimizer, scheduler, device, needs_images=needs_images,
        )

    # Staged unfreezing callback
    training_cfg = config.get("training", {})
    unfreeze_epoch = config.get("unfreeze_epoch")
    unfreeze_lr = training_cfg.get("learning_rate_unfreeze",
                                   training_cfg.get("learning_rate", 5e-5) * 0.1)
    _unfrozen = [False]  # mutable flag for closure

    def staged_unfreeze(epoch, train_result, test_result, verbose):
        if unfreeze_epoch is not None and epoch + 1 == unfreeze_epoch and not _unfrozen[0]:
            print(f"\n*** Epoch {epoch + 1}: Unfreezing encoders (lr={unfreeze_lr})")
            unfreeze_encoders(model, model_name, optimizer, unfreeze_lr)
            _unfrozen[0] = True

    post_epoch_fn = staged_unfreeze if (config.get("freeze_encoders", False) and unfreeze_epoch) else None

    # Train
    num_epochs = training_cfg.get("num_epochs", 50)
    checkpoint_name = config.get("checkpoint_name", "checkpoint")

    if num_epochs > 0:
        print(f"\nStarting training for {num_epochs} epochs...")
        fit_result = trainer.fit(
            dl_train=train_loader,
            dl_test=val_loader,
            num_epochs=num_epochs,
            checkpoints=checkpoint_name,
            early_stopping=training_cfg.get("early_stopping", 45),
            post_epoch_fn=post_epoch_fn,
        )
        print(f"\nTraining complete. Ran {fit_result.num_epochs} epochs.")
    else:
        # Evaluation only — load checkpoint
        checkpoint_file = f"{checkpoint_name}.pt"
        if os.path.isfile(checkpoint_file):
            print(f"Loading checkpoint: {checkpoint_file}")
            saved_state = torch.load(
                checkpoint_file, map_location=device, weights_only=True,
            )
            model.load_state_dict(saved_state["model_state"])
            best = saved_state.get("best_metric", saved_state.get("best_acc", "N/A"))
            print(f"Loaded checkpoint (best_metric: {best})")
        else:
            print(f"Warning: No checkpoint found at {checkpoint_file}")

    # Evaluate
    print("\n" + "=" * 50)
    print("EVALUATION ON TEST SET")
    print("=" * 50)

    if task == "regression":
        metrics = evaluate_regression(model, test_loader, device, needs_images)
        print(f"\nMAE:  {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R²:   {metrics['r2']:.4f}")
    else:
        top1, top2, top3 = evaluate_topk(model, test_loader, device, needs_images)
        print(f"\nTop-1 Accuracy: {top1:.2f}%")
        print(f"Top-2 Accuracy: {top2:.2f}%")
        print(f"Top-3 Accuracy: {top3:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate point cloud classifier")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to JSON config file (see config.json for example)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
