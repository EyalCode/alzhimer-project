"""Smoke tests for the regression pipeline (transfer learning for Alzheimer's)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import pytest


def test_random_split_dataset():
    from util import random_split_dataset
    from datasets import AlzheimerDataset

    ds = AlzheimerDataset('alzhimer_point_clouds', 'alzhimer_src_images')
    train, val, test = random_split_dataset(ds, 0.7, 0.15, seed=42)

    total = len(train) + len(val) + len(test)
    assert total == len(ds), f"Split total {total} != dataset size {len(ds)}"
    assert len(train) > len(val) > 0
    assert len(test) > 0


def test_regression_metrics():
    from util import regression_metrics

    preds = torch.tensor([20.0, 25.0, 15.0, 10.0])
    targets = torch.tensor([22.0, 24.0, 16.0, 8.0])
    m = regression_metrics(preds, targets)

    assert 'mae' in m and 'rmse' in m and 'r2' in m
    assert m['mae'] > 0
    assert m['rmse'] >= m['mae']  # RMSE >= MAE always
    assert m['r2'] > 0  # predictions are correlated


def test_models_num_classes_1():
    from model import PointNet2D, PointNetPlusPlus, PointNetConvNextFusionBase

    pts = torch.randn(2, 2, 1024)
    imgs = torch.randn(2, 3, 224, 224)

    m1 = PointNet2D(num_classes=1)
    assert m1(pts).shape == (2, 1)

    m2 = PointNetPlusPlus(num_classes=1)
    assert m2(pts).shape == (2, 1)

    m3 = PointNetConvNextFusionBase(num_classes=1)
    assert m3(pts, imgs).shape == (2, 1)


def test_regression_trainer_init():
    from model import PointNetPlusPlus
    from train import RegressionTrainer

    model = PointNetPlusPlus(num_classes=1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    trainer = RegressionTrainer(
        model, torch.nn.SmoothL1Loss(), opt, sched,
        device='cpu', needs_images=False,
    )
    assert trainer.higher_is_better is False


def test_freeze_encoders():
    from model import PointNetPlusPlus
    from main import freeze_encoders

    model = PointNetPlusPlus(num_classes=1)
    total_before = sum(1 for p in model.parameters() if p.requires_grad)
    freeze_encoders(model, 'PointNetPlusPlus')
    trainable_after = sum(1 for p in model.parameters() if p.requires_grad)

    assert trainable_after < total_before, "Some params should be frozen"
    assert trainable_after > 0, "Head params should remain trainable"


def test_load_pretrained_encoders_shape_filter():
    """Test that load_pretrained_encoders correctly filters mismatched shapes."""
    from model import PointNetPlusPlus, load_pretrained_encoders

    # Create a "pretrained" model with 250 classes and save it
    pretrained = PointNetPlusPlus(num_classes=250)
    state = {"model_state": pretrained.state_dict()}
    ckpt_path = "test_pretrained_tmp.pt"
    torch.save(state, ckpt_path)

    try:
        # Load into a model with 1 class
        target = PointNetPlusPlus(num_classes=1)
        loaded, skipped = load_pretrained_encoders(target, ckpt_path)

        assert len(skipped) > 0, "Should skip final layer params"
        assert "fc3.weight" in skipped
        assert "fc3.bias" in skipped
        assert len(loaded) > len(skipped), "Most params should load"
    finally:
        os.remove(ckpt_path)


def test_build_loss_fn_regression():
    from main import build_loss_fn

    config_reg = {"task": "regression", "training": {"loss": "SmoothL1"}}
    loss = build_loss_fn(config_reg)
    assert isinstance(loss, torch.nn.SmoothL1Loss)

    config_mse = {"task": "regression", "training": {"loss": "MSE"}}
    loss2 = build_loss_fn(config_mse)
    assert isinstance(loss2, torch.nn.MSELoss)


def test_alzheimer_config_loads():
    import json
    with open('alzheimer_config.json') as f:
        cfg = json.load(f)
    assert cfg['task'] == 'regression'
    assert cfg['num_classes'] == 1
    assert cfg['data']['point_clouds_dir'] == 'alzhimer_point_clouds'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
