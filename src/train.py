"""
Training framework: abstract Trainer base class and PointNet2dClassifierTrainer.

The Trainer provides a fit/train_epoch/test_epoch loop with checkpointing and
early stopping. PointNet2dClassifierTrainer handles both point-only and fusion
model training via the needs_images flag.
"""

import abc
import os
import sys

import torch
import tqdm
from typing import Any, Callable, List, NamedTuple
from pathlib import Path
from torch.utils.data import DataLoader


class BatchResult(NamedTuple):
    """Result of training/testing a single batch."""
    loss: float
    num_correct: int


class EpochResult(NamedTuple):
    """Result of a single epoch: per-batch losses and overall accuracy."""
    losses: List[float]
    accuracy: float


class FitResult(NamedTuple):
    """Result of fitting a model over multiple epochs."""
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]


class Trainer(abc.ABC):
    """Abstract base class for model training.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch / test_epoch)
    - Single batch (train_batch / test_batch) — implemented by subclasses
    """

    def __init__(self, model, loss_fn, optimizer, scheduler, device="cpu", higher_is_better=True):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.higher_is_better = higher_is_better
        model.to(self.device)

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every=1,
        post_epoch_fn=None,
        **kw,
    ) -> FitResult:
        """Train the model for multiple epochs with validation.

        Args:
            dl_train: Training dataloader.
            dl_test: Validation dataloader.
            num_epochs: Number of epochs to train.
            checkpoints: Checkpoint filename (without .pt extension). None to skip.
            early_stopping: Stop after this many epochs without improvement. None to disable.
            print_every: Print progress every N epochs.
            post_epoch_fn: Optional callback(epoch, train_result, test_result, verbose).

        Returns:
            FitResult with training history.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_metric = None
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f"{checkpoints}.pt"
            checkpoint_dir = os.path.dirname(checkpoint_filename)
            if checkpoint_dir:
                Path(checkpoint_dir).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f"*** Loading checkpoint file {checkpoint_filename}")
                saved_state = torch.load(
                    checkpoint_filename, map_location=self.device, weights_only=True
                )
                best_metric = saved_state.get("best_metric", saved_state.get("best_acc", best_metric))
                epochs_without_improvement = saved_state.get(
                    "ewi", epochs_without_improvement
                )
                self.model.load_state_dict(saved_state["model_state"])

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = (epoch % print_every == 0) or (epoch == num_epochs - 1)
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)

            train_result = self.train_epoch(dl_train=dl_train, verbose=verbose, **kw)
            train_acc.append(train_result.accuracy)
            train_loss += train_result.losses

            test_result = self.test_epoch(dl_test, verbose=verbose, **kw)
            test_loss += test_result.losses
            test_acc.append(test_result.accuracy)

            actual_num_epochs += 1
            self.scheduler.step()

            improved = (
                best_metric is None
                or (test_result.accuracy > best_metric if self.higher_is_better
                    else test_result.accuracy < best_metric)
            )
            if improved:
                epochs_without_improvement = 0
                best_metric = test_result.accuracy
                save_checkpoint = True
            else:
                epochs_without_improvement += 1
                if early_stopping is not None and epochs_without_improvement == early_stopping:
                    break

            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(
                    best_metric=best_metric,
                    ewi=epochs_without_improvement,
                    model_state=self.model.state_dict(),
                )
                torch.save(saved_state, checkpoint_filename)
                print(f"*** Saved checkpoint {checkpoint_filename} at epoch {epoch+1}")

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """Train once over a training set (single epoch)."""
        self.model.train(True)
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """Evaluate model once over a test set (single epoch)."""
        self.model.train(False)
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """Run a single training batch: forward, loss, backward, optimize."""
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """Run a single test batch: forward and loss (no gradients)."""
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
        dl: DataLoader,
        forward_fn: Callable[[Any], BatchResult],
        verbose=True,
        max_batches=None,
    ) -> EpochResult:
        """Run forward_fn on all batches from the dataloader with progress bar."""
        losses = []
        num_correct = 0

        num_samples = len(dl.dataset)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None and max_batches < num_batches:
            num_batches = max_batches
            num_samples = num_batches * dl.batch_size

        pbar_file = sys.stdout if verbose else open(os.devnull, "w")

        try:
            pbar_name = forward_fn.__name__
            with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
                dl_iter = iter(dl)
                for batch_idx in range(num_batches):
                    data = next(dl_iter)
                    batch_res = forward_fn(data)

                    pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                    pbar.update()

                    losses.append(batch_res.loss)
                    num_correct += batch_res.num_correct

                avg_loss = sum(losses) / num_batches
                accuracy = 100.0 * num_correct / num_samples
                pbar.set_description(
                    f"{pbar_name} "
                    f"(Avg. Loss {avg_loss:.3f}, "
                    f"Accuracy {accuracy:.1f})"
                )
        finally:
            if pbar_file is not sys.stdout:
                pbar_file.close()

        return EpochResult(losses=losses, accuracy=accuracy)


class PointNet2dClassifierTrainer(Trainer):
    """Trainer for point cloud classifiers (point-only and fusion models).

    Args:
        model: The classifier model.
        loss_fn: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to train on.
        needs_images: If True, expects batches of (points, labels, images)
                      and passes both points and images to model.forward().
                      If False, expects (points, labels) batches.
    """

    def __init__(self, model, loss_fn, optimizer, scheduler, device="cuda", needs_images=False):
        super().__init__(model, loss_fn, optimizer, scheduler, device, higher_is_better=True)
        self.needs_images = needs_images

    def train_batch(self, batch) -> BatchResult:
        if self.needs_images:
            x, y, imgs = batch
            x, y, imgs = x.to(self.device), y.to(self.device), imgs.to(self.device)
        else:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.model(x, imgs) if self.needs_images else self.model(x)

        loss = self.loss_fn(outputs, y)
        loss.backward()
        self.optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct_predictions = (predicted == y).sum().item()

        return BatchResult(loss.item(), correct_predictions)

    def test_batch(self, batch) -> BatchResult:
        if self.needs_images:
            x, y, imgs = batch
            x, y, imgs = x.to(self.device), y.to(self.device), imgs.to(self.device)
        else:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            outputs = self.model(x, imgs) if self.needs_images else self.model(x)
            loss = self.loss_fn(outputs, y)

            _, predicted = torch.max(outputs.data, 1)
            correct_predictions = (predicted == y).sum().item()

        return BatchResult(loss.item(), correct_predictions)


class RegressionTrainer(Trainer):
    """Trainer for regression models predicting continuous MOCA scores.

    Uses MAE as the tracking metric (lower is better) for checkpointing
    and early stopping. Overrides train_epoch/test_epoch to compute MAE
    instead of classification accuracy.

    Args:
        model: The regression model.
        loss_fn: Loss function (e.g. SmoothL1Loss, MSELoss).
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to train on.
        needs_images: If True, expects (points, labels, images) batches.
    """

    def __init__(self, model, loss_fn, optimizer, scheduler, device="cuda", needs_images=False):
        super().__init__(model, loss_fn, optimizer, scheduler, device, higher_is_better=False)
        self.needs_images = needs_images

    def train_batch(self, batch) -> BatchResult:
        if self.needs_images:
            x, y, imgs = batch
            x, y, imgs = x.to(self.device), y.to(self.device), imgs.to(self.device)
        else:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(x, imgs) if self.needs_images else self.model(x)
        outputs = outputs.squeeze(-1)  # (B, 1) -> (B,)
        loss = self.loss_fn(outputs, y)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            sum_ae = torch.abs(outputs - y).sum().item()

        return BatchResult(loss.item(), sum_ae)

    def test_batch(self, batch) -> BatchResult:
        if self.needs_images:
            x, y, imgs = batch
            x, y, imgs = x.to(self.device), y.to(self.device), imgs.to(self.device)
        else:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            outputs = self.model(x, imgs) if self.needs_images else self.model(x)
            outputs = outputs.squeeze(-1)
            loss = self.loss_fn(outputs, y)
            sum_ae = torch.abs(outputs - y).sum().item()

        return BatchResult(loss.item(), sum_ae)

    def train_epoch(self, dl_train, **kw):
        self.model.train(True)
        return self._run_epoch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test, **kw):
        self.model.train(False)
        return self._run_epoch(dl_test, self.test_batch, **kw)

    def _run_epoch(self, dl, forward_fn, verbose=True, max_batches=None):
        losses = []
        sum_ae_total = 0.0
        num_samples = len(dl.dataset)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None and max_batches < num_batches:
            num_batches = max_batches
            num_samples = num_batches * dl.batch_size

        pbar_file = sys.stdout if verbose else open(os.devnull, "w")
        try:
            pbar_name = forward_fn.__name__
            with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
                dl_iter = iter(dl)
                for _ in range(num_batches):
                    data = next(dl_iter)
                    batch_res = forward_fn(data)
                    pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                    pbar.update()
                    losses.append(batch_res.loss)
                    sum_ae_total += batch_res.num_correct

                mae = sum_ae_total / num_samples
                avg_loss = sum(losses) / num_batches
                pbar.set_description(
                    f"{pbar_name} "
                    f"(Avg. Loss {avg_loss:.3f}, "
                    f"MAE {mae:.2f})"
                )
        finally:
            if pbar_file is not sys.stdout:
                pbar_file.close()

        return EpochResult(losses=losses, accuracy=mae)
