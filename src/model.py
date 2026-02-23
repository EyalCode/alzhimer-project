"""
Model architectures for point cloud and multimodal sketch classification.

Models:
    - TNet: Spatial transformer network (used by PointNet2D)
    - PointNet2D: Baseline 2D PointNet classifier
    - PointNetPlusPlus: Hierarchical PointNet++ with multi-scale grouping
    - PointNetResNetFusion: PointNet++ + ResNet50 late fusion
    - PointNetConvNextFusion: PointNet++ + ConvNeXt-Tiny late fusion
    - PointNetConvNextFusionBase: PointNet++ + ConvNeXt-Base late fusion (best model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    resnet50, ResNet50_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
)
from pointnet2_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg


def load_pretrained_encoders(model, checkpoint_path, device="cpu"):
    """Load encoder weights from a classification checkpoint, skipping mismatched head layers.

    Works with all model types. Only the final output layer (which changes shape
    when num_classes differs) is skipped; all encoder and intermediate head params load.

    Args:
        model: Target model (e.g., any model with num_classes=1 for regression).
        checkpoint_path: Path to checkpoint (.pt file, without extension is also tried).
        device: Device to load weights to.

    Returns:
        (loaded_keys, skipped_keys) — sets of parameter names.
    """
    # Try with and without .pt extension
    import os
    path = checkpoint_path
    if not os.path.isfile(path):
        path = f"{checkpoint_path}.pt"

    saved = torch.load(path, map_location=device, weights_only=True)
    pretrained_dict = saved["model_state"]

    # Filter out params with shape mismatch (the final classification layer)
    model_dict = model.state_dict()
    compatible = {}
    skipped = set()
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.add(k)

    model.load_state_dict(compatible, strict=False)
    return set(compatible.keys()), skipped


# ---------------------------------------------------------------------------
# Spatial Transformer Network
# ---------------------------------------------------------------------------

class TNet(nn.Module):
    """Spatial transformer network that predicts a k x k transformation matrix.

    Input:  (B, k, N) point features
    Output: (B, k, k) transformation matrix
    """

    def __init__(self, k=64):
        super().__init__()

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).flatten().unsqueeze(0).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# ---------------------------------------------------------------------------
# Point-Only Models
# ---------------------------------------------------------------------------

class PointNet2D(nn.Module):
    """Baseline 2D PointNet classifier.

    Input:  (B, 2, N) — 2D point cloud, channels-first
    Output: (B, num_classes) — raw logits
    """

    def __init__(self, num_classes=10, dropout=0.33):
        super().__init__()

        self.feature_transform = TNet(k=64)

        # First shared MLP
        self.conv1 = nn.Conv1d(2, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)

        # Second shared MLP
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, 2, N)

        # First shared MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transformation
        trans_feat = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)

        # Second shared MLP
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Max pooling → global feature
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # Classification head
        x = F.relu(self.bn6(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class PointNetPlusPlus(nn.Module):
    """Hierarchical PointNet++ with multi-scale grouping (MSG).

    SA layers:
        SA1: 512 centroids, radii [0.2, 0.4, 0.8], groups [16, 32, 128] → 320-dim
        SA2: 128 centroids, radii [0.2, 0.4, 0.8], groups [32, 64, 128] → 640-dim
        SA3: Global aggregation → 1024-dim

    Input:  (B, 2, N) — 2D point cloud, channels-first
    Output: (B, num_classes) — raw logits
    """

    def __init__(self, num_classes=250, normal_channel=False):
        super().__init__()
        in_channel = 2 if normal_channel else 0
        self.normal_channel = normal_channel

        self.sa1 = PointNetSetAbstractionMsg(
            512, [0.2, 0.4, 0.8], [16, 32, 128], in_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstraction(None, None, None, 640, [256, 512, 1024], True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xy):
        B, _, _ = xy.shape

        if self.normal_channel:
            norm = xy[:, 2:, :]
            xy = xy[:, :2, :]
        else:
            norm = None

        l1_xy, l1_points = self.sa1(xy, norm)
        l2_xy, l2_points = self.sa2(l1_xy, l1_points)
        l3_xy, l3_points = self.sa3(l2_xy, l2_points)

        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x


# ---------------------------------------------------------------------------
# Fusion Models (PointNet++ + CNN backbone)
# ---------------------------------------------------------------------------

class PointNetResNetFusion(nn.Module):
    """Late fusion: PointNet++ (1024-dim) + ResNet50 (2048-dim) → MLP → classes.

    Input:  xy (B, 2, N), img (B, 3, 224, 224)
    Output: (B, num_classes) — raw logits
    """

    def __init__(self, num_classes=250, normal_channel=False):
        super().__init__()
        in_channel = 2 if normal_channel else 0
        self.normal_channel = normal_channel

        # PointNet++ encoder
        self.sa1 = PointNetSetAbstractionMsg(
            512, [0.2, 0.4, 0.8], [16, 32, 128], in_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstraction(None, None, None, 640, [256, 512, 1024], True)

        # CNN encoder
        self.cnn = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()

        # Fusion head: 1024 (PointNet++) + 2048 (ResNet50) = 3072
        self.fc1 = nn.Linear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xy, img):
        B, _, _ = xy.shape

        if self.normal_channel:
            norm = xy[:, 2:, :]
            xy = xy[:, :2, :]
        else:
            norm = None

        l1_xy, l1_points = self.sa1(xy, norm)
        l2_xy, l2_points = self.sa2(l1_xy, l1_points)
        l3_xy, l3_points = self.sa3(l2_xy, l2_points)
        point_feats = l3_points.view(B, 1024)

        img_feats = self.cnn(img)
        combined = torch.cat((point_feats, img_feats), dim=1)

        x = self.drop1(F.relu(self.bn1(self.fc1(combined))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x


class PointNetConvNextFusion(nn.Module):
    """Late fusion: PointNet++ (1024-dim) + ConvNeXt-Tiny (768-dim) → MLP → classes.

    Input:  xy (B, 2, N), img (B, 3, 224, 224)
    Output: (B, num_classes) — raw logits
    """

    def __init__(self, num_classes=250, normal_channel=False):
        super().__init__()
        in_channel = 2 if normal_channel else 0
        self.normal_channel = normal_channel

        # PointNet++ encoder
        self.sa1 = PointNetSetAbstractionMsg(
            512, [0.2, 0.4, 0.8], [16, 32, 128], in_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstraction(None, None, None, 640, [256, 512, 1024], True)

        # CNN encoder
        self.cnn = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.cnn.classifier = nn.Identity()

        # Fusion head: 1024 (PointNet++) + 768 (ConvNeXt-Tiny) = 1792
        self.fc1 = nn.Linear(1792, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xy, img):
        B, _, _ = xy.shape

        if self.normal_channel:
            norm = xy[:, 2:, :]
            xy = xy[:, :2, :]
        else:
            norm = None

        l1_xy, l1_points = self.sa1(xy, norm)
        l2_xy, l2_points = self.sa2(l1_xy, l1_points)
        l3_xy, l3_points = self.sa3(l2_xy, l2_points)
        point_feats = l3_points.view(B, 1024)

        img_feats = self.cnn(img).flatten(1)
        combined = torch.cat((point_feats, img_feats), dim=1)

        x = self.drop1(F.relu(self.bn1(self.fc1(combined))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x


class PointNetConvNextFusionBase(nn.Module):
    """Late fusion: PointNet++ (1024-dim) + ConvNeXt-Base (1024-dim) → deep MLP → classes.

    Best performing model: 82.8% Top-1, 94% Top-3 on TU-Berlin (250 classes).

    Input:  point_input (B, 2, N), img_input (B, 3, 224, 224)
    Output: (B, num_classes) — raw logits
    """

    def __init__(self, num_classes=250, normal_channel=False):
        super().__init__()
        in_channel = 2 if normal_channel else 0
        self.normal_channel = normal_channel

        # PointNet++ encoder
        self.sa1 = PointNetSetAbstractionMsg(
            512, [0.2, 0.4, 0.8], [16, 32, 128], in_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstraction(None, None, None, 640, [256, 512, 1024], True)
        self.pointnet_output_dim = 1024

        # CNN encoder
        self.cnn = models.convnext_base(weights='DEFAULT')
        self.cnn.classifier = nn.Identity()
        self.cnn_output_dim = 1024

        # Deep fusion head: 1024 + 1024 = 2048
        fusion_input_dim = self.pointnet_output_dim + self.cnn_output_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes),
        )

    def forward(self, point_input, img_input):
        xy = point_input
        B, _, _ = xy.shape

        if self.normal_channel:
            norm = xy[:, 2:, :]
            xy = xy[:, :2, :]
        else:
            norm = None

        l1_xy, l1_points = self.sa1(xy, norm)
        l2_xy, l2_points = self.sa2(l1_xy, l1_points)
        l3_xy, l3_points = self.sa3(l2_xy, l2_points)
        point_feats = l3_points.view(B, 1024)

        img_feats = self.cnn(img_input)
        img_feats = img_feats.view(img_feats.size(0), -1)

        combined = torch.cat((point_feats, img_feats), dim=1)
        output = self.fusion_fc(combined)

        return output
