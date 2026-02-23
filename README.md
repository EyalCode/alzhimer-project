# Alzheimer's Detection from Self-Portraits

Point-based deep learning for sketch recognition and cognitive decline detection. The project converts 2D sketches into point clouds (1024 points per image) and classifies them using PointNet-family architectures, optionally fused with CNN image features.

Two tasks are supported:

- **Classification** (Part A): 250-class sketch recognition on the [TU-Berlin Sketch Dataset](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) — used to validate architectures.
- **Regression** (Part B): Predict MoCA cognitive scores from self-portrait sketches drawn by elderly individuals (Alzheimer's detection).

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd image-recognition-project

# 2. Create the conda environment
conda env create -f env.yml
conda activate image-recognition-project

# 3. Prepare your data (see "Data Preparation" below)

# 4. Train a model
python src/main.py --config config.json
```

## Project Structure

```
image-recognition-project/
├── config.json                  # Example config: TU-Berlin classification (250 classes)
├── alzheimer_config.json        # Example config: Alzheimer's regression (MoCA scores)
├── env.yml                      # Conda environment (Python 3.13, PyTorch 2.5.1)
├── generate_results_csv.py      # Parse experiment logs into a summary CSV
│
├── src/
│   ├── main.py                  # Entry point — loads config, builds model, trains/evaluates
│   ├── model.py                 # All model architectures
│   ├── train.py                 # Trainer classes (classification + regression)
│   ├── datasets.py              # Dataset loaders (TUBerlinDataset, AlzheimerDataset)
│   ├── util.py                  # Augmentation, data splitting, metrics, seeding
│   └── pointnet2_utils.py       # Low-level PointNet++ ops (FPS, ball query)
│
├── scripts/
│   └── prepare_dataset.py       # Convert raw Alzheimer's images → point clouds
│
├── tests/
│   ├── test_alzheimer_dataset.py
│   └── test_regression_smoke.py
│
├── splits/                      # Predefined train/val/test splits for Alzheimer's
│   ├── data_2026_train_list_split_1.txt
│   ├── data_2026_validation_list_split_1.txt
│   └── data_2026_test_list_split_1.txt
│
├── preprocessed_point_clouds/   # TU-Berlin point clouds (not in repo, see Data Preparation)
└── preprocessed_point_clouds.zip # Compressed TU-Berlin point clouds
```

## Data Preparation

### TU-Berlin Dataset (Classification)

The training pipeline expects two directories at the project root:

| Directory | Contents | Required By |
|---|---|---|
| `preprocessed_point_clouds/` | One `.pt` file per sketch (`00000.pt`, `00001.pt`, ...). Each contains `{'point_cloud': Tensor(2, 1024), 'label': int}`. | All models |
| `sketches_png/png/` | Original sketch images organized by class: `<class_name>/*.png` | Fusion models only |

If `preprocessed_point_clouds.zip` is included in the repo, unzip it:

```bash
unzip preprocessed_point_clouds.zip
```

### Alzheimer's Dataset (Regression)

**Step 1.** Place your raw data files at the project root:

- `data_all_files_2026.txt` — Label file with one entry per line: `<MoCA_score>  <Collection/Filename.png>`
- `alzheimer_2026_all_blended_512/` — Source images organized as `<Collection>/<Filename>.png`

**Step 2.** Run the preprocessing script:

```bash
python scripts/prepare_dataset.py
```

This produces two output directories:

| Output Directory | Contents |
|---|---|
| `alzhimer_point_clouds/` | Point clouds as `.pt` files: `<Collection>/<stem>.pt`, each containing `{'point_cloud': Tensor(1024, 2), 'label': int}` |
| `alzhimer_src_images/` | Copies of the source images: `<Collection>/<Filename>.png` |

The script extracts "ink" pixels from each grayscale image, samples 1024 points using stratified grid sampling, and normalizes coordinates to [-1, 1].

## Configuration

All training is driven by a JSON config file passed via `--config`:

```bash
python src/main.py --config config.json
```

### Classification Config Example

```jsonc
{
    "model": "PointNetConvNextFusionBase",  // Which model to use (see "Available Models")
    "num_classes": 250,                     // Number of output classes
    "use_normals": false,                   // Add engineered features (x², y², xy)

    "data": {
        "point_clouds_dir": "preprocessed_point_clouds",
        "images_dir": "sketches_png/png"    // Required for fusion models
    },

    "training": {
        "batch_size": 64,
        "num_epochs": 50,                   // Set to 0 for evaluation only
        "early_stopping": 45,               // Patience: epochs without improvement
        "learning_rate": 5e-05,             // LR for PointNet++ and classification head
        "learning_rate_cnn": 5e-06,         // LR for CNN backbone (fusion models)
        "weight_decay": 0.1,
        "label_smoothing": 0.1,
        "scheduler": "CosineAnnealingLR",
        "scheduler_T_max": 50,
        "scheduler_eta_min": 7e-08
    },

    "augmentation": {
        "rotation_range": 90,               // Random rotation ± degrees
        "scale_range": [0.5, 1.5],
        "translation_range": 0.2,
        "jitter_std": 0.008,                // Gaussian noise std dev
        "point_dropout_rate": 0.2           // Fraction of points randomly zeroed
    },

    "checkpoint_name": "TUBerlin_checkpoints",  // Saved as <name>.pt
    "seed": 42
}
```

### Regression Config Example

```jsonc
{
    "task": "regression",                       // "classification" (default) or "regression"
    "model": "PointNetConvNextFusionBase",
    "num_classes": 1,                           // Single output (MoCA score)
    "use_normals": false,

    "pretrained_checkpoint": "TUBerlin_checkpoints",  // Load encoder weights from this checkpoint
    "freeze_encoders": false,                         // Freeze encoder layers (head stays trainable)

    "data": {
        "point_clouds_dir": "alzhimer_point_clouds",
        "images_dir": "alzhimer_src_images",
        "train_fraction": 0.7,                  // For random split
        "val_fraction": 0.15
        // Or use predefined splits:
        // "split_files": {
        //     "train": "splits/data_2026_train_list_split_1.txt",
        //     "val": "splits/data_2026_validation_list_split_1.txt",
        //     "test": "splits/data_2026_test_list_split_1.txt"
        // }
    },

    "training": {
        "batch_size": 16,
        "num_epochs": 100,
        "early_stopping": 30,
        "learning_rate": 1e-4,
        "learning_rate_cnn": 1e-6,
        "weight_decay": 0.05,
        "loss": "SmoothL1",                     // "SmoothL1" or "MSE"
        "scheduler": "CosineAnnealingLR",
        "scheduler_T_max": 100,
        "scheduler_eta_min": 1e-7
    },

    "augmentation": {
        "rotation_range": 50,
        "scale_range": [0.8, 1.2],
        "translation_range": 0.15,
        "jitter_std": 0.003,
        "point_dropout_rate": 0.1
    },

    "checkpoint_name": "alzheimer_regression",
    "seed": 42
}
```

### Full Config Reference

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | — | Model architecture (see Available Models) |
| `num_classes` | int | 250 | Output size (250 for TU-Berlin, 1 for regression) |
| `use_normals` | bool | false | Append engineered features (x², y², xy) to each point |
| `task` | string | `"classification"` | `"classification"` or `"regression"` |
| `device` | string | `"auto"` | `"auto"`, `"cuda"`, or `"cpu"` |
| `seed` | int | 42 | Random seed for reproducibility |
| `checkpoint_name` | string | `"checkpoint"` | Checkpoint filename (saved as `<name>.pt`) |
| **Data** ||||
| `data.point_clouds_dir` | string | — | Path to `.pt` point cloud files |
| `data.images_dir` | string | — | Path to sketch images (required for fusion models) |
| `data.train_fraction` | float | 0.7 | Training split fraction (random split, Alzheimer's only) |
| `data.val_fraction` | float | 0.15 | Validation split fraction (random split, Alzheimer's only) |
| `data.split_files.train` | string | — | Path to predefined training split file |
| `data.split_files.val` | string | — | Path to predefined validation split file |
| `data.split_files.test` | string | — | Path to predefined test split file |
| **Training** ||||
| `training.batch_size` | int | 64 | Batch size |
| `training.num_epochs` | int | 50 | Training epochs (0 = evaluation only) |
| `training.early_stopping` | int | 45 | Stop after N epochs without metric improvement |
| `training.learning_rate` | float | 5e-5 | LR for PointNet++ encoder and classification/regression head |
| `training.learning_rate_cnn` | float | 5e-6 | LR for CNN backbone (fusion models only, typically 10x lower) |
| `training.learning_rate_unfreeze` | float | lr × 0.1 | LR applied when encoders are unfrozen mid-training |
| `training.weight_decay` | float | 0.1 | AdamW weight decay |
| `training.label_smoothing` | float | 0.1 | Label smoothing for CrossEntropyLoss (classification only) |
| `training.loss` | string | `"SmoothL1"` | Loss function: `"SmoothL1"` or `"MSE"` (regression only) |
| `training.scheduler` | string | `"CosineAnnealingLR"` | LR scheduler: `"CosineAnnealingLR"` or `"StepLR"` |
| `training.scheduler_T_max` | int | 50 | CosineAnnealingLR: period in epochs |
| `training.scheduler_eta_min` | float | 7e-8 | CosineAnnealingLR: minimum LR |
| `training.scheduler_step_size` | int | 30 | StepLR: decay every N epochs |
| `training.scheduler_gamma` | float | 0.9 | StepLR: multiply LR by this factor |
| **Augmentation** ||||
| `augmentation.rotation_range` | int | 90 | Random rotation ± degrees |
| `augmentation.scale_range` | [float, float] | [0.5, 1.5] | Random scale bounds |
| `augmentation.translation_range` | float | 0.2 | Max translation offset |
| `augmentation.jitter_std` | float | 0.008 | Gaussian noise standard deviation |
| `augmentation.point_dropout_rate` | float | 0.2 | Max fraction of points to randomly zero out |
| **Transfer Learning** ||||
| `pretrained_checkpoint` | string | — | Load encoder weights from this checkpoint file (without `.pt`) |
| `freeze_encoders` | bool | false | Freeze encoder parameters (only train the head) |
| `unfreeze_epoch` | int | — | Epoch at which to unfreeze encoders for fine-tuning |
| `moca_translation` | bool | false | Apply MoCA score transformation: `label = label / 3.0 + 12.0` |

## Available Models

| Model name | Description | Needs Images | TU-Berlin Top-1 |
|---|---|---|---|
| `PointNet2D` | Baseline 2D PointNet with spatial transformer | No | 56.1% |
| `PointNetPlusPlus` | Hierarchical PointNet++ with multi-scale grouping | No | 67-68% |
| `PointNetResNetFusion` | PointNet++ + ResNet50 (late fusion) | Yes | 75.0% |
| `PointNetConvNextFusion` | PointNet++ + ConvNeXt-Tiny (late fusion) | Yes | 82.0% |
| `PointNetConvNextFusionBase` | PointNet++ + ConvNeXt-Base (late fusion) | Yes | **82.8%** |

**Point-only models** (`PointNet2D`, `PointNetPlusPlus`) only need point cloud data.
**Fusion models** combine point cloud and image features — they require `data.images_dir` in the config.

## Training

### Classification (TU-Berlin)

1. Prepare data (see Data Preparation above).
2. Edit `config.json` — set `model`, adjust hyperparameters as needed.
3. Run:

```bash
python src/main.py --config config.json
```

The trainer logs accuracy each epoch, saves the best checkpoint, and stops early if accuracy doesn't improve for `early_stopping` epochs.

### Regression (Alzheimer's)

1. Run `scripts/prepare_dataset.py` to generate point clouds.
2. Edit `alzheimer_config.json` — key fields: `"task": "regression"`, `"num_classes": 1`.
3. Run:

```bash
python src/main.py --config alzheimer_config.json
```

Regression metrics reported: MAE, RMSE, and R².

### Transfer Learning

Train a classification model on TU-Berlin first, then fine-tune on the Alzheimer's dataset:

1. Train on TU-Berlin:
   ```bash
   python src/main.py --config config.json
   ```
   This saves a checkpoint (e.g., `TUBerlin_checkpoints.pt`).

2. In your regression config, set:
   ```json
   "pretrained_checkpoint": "TUBerlin_checkpoints"
   ```
   The encoder weights will be loaded from this checkpoint. The classification head is discarded and replaced with a new regression head.

3. Optionally freeze encoders and unfreeze later:
   ```json
   "freeze_encoders": true,
   "unfreeze_epoch": 30
   ```
   This trains only the regression head for the first 30 epochs, then unfreezes all layers.

### Evaluation Only

Set `num_epochs` to 0 in your config. The trainer will load the checkpoint and evaluate on the test set without training:

```bash
# In config.json: "training": { "num_epochs": 0, ... }
python src/main.py --config config.json
```

## Results (TU-Berlin, 250 classes)

| Model | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| PointNet2D (baseline) | 56.1% | — | — |
| PointNet++ | 67-68% | — | — |
| PointNet++ + ResNet50 | 75.0% | — | — |
| PointNet++ + ConvNeXt-Tiny | 82.0% | — | — |
| **PointNet++ + ConvNeXt-Base** | **82.8%** | **92.0%** | **94.0%** |

## Tests

```bash
python -m pytest tests/
```
