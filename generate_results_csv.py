"""
Parse all experiment results from saved_results/ into a single CSV.

Usage:
    python generate_results_csv.py [--output PATH]

Walks saved_results/ looking for directories with config_used.json + slurm.out,
extracts configuration and metrics, and writes results_summary.csv.
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path

SAVED_RESULTS_DIR = Path("saved_results")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m|\x1b\[[\d;]*[A-Za-z]|\r")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def parse_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        cfg = json.load(f)

    training = cfg.get("training", {})
    data = cfg.get("data", {})

    pc_dir = data.get("point_clouds_dir", "")
    dataset = "india" if "india" in pc_dir.lower() else "all-data"
    split_method = "predefined" if "split_files" in data else "random"

    return {
        "model": cfg.get("model", ""),
        "dataset": dataset,
        "moca_translation": cfg.get("moca_translation", False),
        "loss_function": training.get("loss", ""),
        "batch_size": training.get("batch_size", ""),
        "lr": training.get("learning_rate", ""),
        "lr_cnn": training.get("learning_rate_cnn", ""),
        "lr_unfreeze": training.get("learning_rate_unfreeze", ""),
        "weight_decay": training.get("weight_decay", ""),
        "unfreeze_epoch": cfg.get("unfreeze_epoch", ""),
        "early_stopping_patience": training.get("early_stopping", ""),
        "num_epochs_config": training.get("num_epochs", 100),
        "split_method": split_method,
        "augmentation": cfg.get("augmentation", False),
        "seed": cfg.get("seed", ""),
    }


def parse_slurm(slurm_path: Path, num_epochs_config: int) -> dict:
    with open(slurm_path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    text = strip_ansi(raw)

    result = {
        "job_id": "",
        "node": "",
        "started": "",
        "epochs_trained": "",
        "early_stopped": "",
        "test_mae": "",
        "test_rmse": "",
        "test_r2": "",
        "best_val_mae": "",
        "best_val_loss": "",
        "last_train_mae": "",
        "last_train_loss": "",
    }

    # Job metadata
    m = re.search(r"Job ID\s*:\s*(\S+)", text)
    if m:
        result["job_id"] = m.group(1)
    m = re.search(r"Node\s*:\s*(\S+)", text)
    if m:
        result["node"] = m.group(1)
    m = re.search(r"Started\s*:\s*(.+)", text)
    if m:
        result["started"] = m.group(1).strip()

    # Epochs trained
    m = re.search(r"Training complete\. Ran (\d+) epochs", text)
    if m:
        epochs = int(m.group(1))
        result["epochs_trained"] = epochs
        result["early_stopped"] = epochs < num_epochs_config

    # Final test metrics (from EVALUATION ON TEST SET block)
    m = re.search(r"EVALUATION ON TEST SET.*?MAE:\s*([\d.]+).*?RMSE:\s*([\d.]+).*?R..?:\s*([-\d.]+)", text, re.DOTALL)
    if m:
        result["test_mae"] = float(m.group(1))
        result["test_rmse"] = float(m.group(2))
        result["test_r2"] = float(m.group(3))

    # Validation metrics during training (logged as test_batch, but actually val set)
    # Pattern: test_batch (Avg. Loss <loss>, MAE <mae>)
    val_matches = re.findall(r"test_batch \(Avg\. Loss ([\d.]+), MAE ([\d.]+)\)", text)
    if val_matches:
        best_val_mae = float("inf")
        best_val_loss = None
        for loss_str, mae_str in val_matches:
            mae = float(mae_str)
            if mae < best_val_mae:
                best_val_mae = mae
                best_val_loss = float(loss_str)
        result["best_val_mae"] = round(best_val_mae, 3)
        result["best_val_loss"] = round(best_val_loss, 3)

    # Last training metrics
    train_matches = re.findall(r"train_batch \(Avg\. Loss ([\d.]+), MAE ([\d.]+)\)", text)
    if train_matches:
        last_loss, last_mae = train_matches[-1]
        result["last_train_loss"] = round(float(last_loss), 3)
        result["last_train_mae"] = round(float(last_mae), 3)

    return result


def collect_results(results_dir: Path) -> list[dict]:
    rows = []
    for root, dirs, files in os.walk(results_dir):
        root_path = Path(root)
        if "config_used.json" in files and "slurm.out" in files:
            config_path = root_path / "config_used.json"
            slurm_path = root_path / "slurm.out"

            run_name = root_path.name
            experiment_group = root_path.parent.name

            try:
                config_data = parse_config(config_path)
            except Exception as e:
                print(f"  WARNING: Failed to parse {config_path}: {e}")
                continue

            try:
                slurm_data = parse_slurm(slurm_path, config_data["num_epochs_config"])
            except Exception as e:
                print(f"  WARNING: Failed to parse {slurm_path}: {e}")
                continue

            row = {"experiment_group": experiment_group, "run_name": run_name}
            row.update(config_data)
            row.update(slurm_data)
            del row["num_epochs_config"]  # internal use only
            rows.append(row)

    # Sort by dataset, model, loss_function
    rows.sort(key=lambda r: (r["dataset"], r["model"], r["loss_function"]))
    return rows


CSV_COLUMNS = [
    "experiment_group",
    "run_name",
    "model",
    "dataset",
    "loss_function",
    "moca_translation",
    "epochs_trained",
    "early_stopped",
    "test_mae",
    "test_rmse",
    "test_r2",
    "best_val_mae",
    "best_val_loss",
    "last_train_mae",
    "last_train_loss",
    "batch_size",
    "lr",
    "lr_cnn",
    "lr_unfreeze",
    "weight_decay",
    "unfreeze_epoch",
    "early_stopping_patience",
    "split_method",
    "augmentation",
    "seed",
    "job_id",
    "node",
    "started",
]


def print_summary(rows: list[dict]):
    print(f"\n{'='*80}")
    print(f"  Results Summary — {len(rows)} runs")
    print(f"{'='*80}\n")

    header = f"{'Model':<30} {'Dataset':<10} {'Loss':<10} {'MoCA Tr.':<9} {'Test MAE':<10} {'Test RMSE':<10} {'Test R²':<10} {'Ep.':<5}"
    print(header)
    print("-" * len(header))
    for r in rows:
        moca = "Yes" if r["moca_translation"] else "No"
        mae = f"{r['test_mae']:.2f}" if isinstance(r["test_mae"], (int, float)) else "N/A"
        rmse = f"{r['test_rmse']:.2f}" if isinstance(r["test_rmse"], (int, float)) else "N/A"
        r2 = f"{r['test_r2']:.4f}" if isinstance(r["test_r2"], (int, float)) else "N/A"
        ep = str(r["epochs_trained"]) if r["epochs_trained"] else "N/A"
        print(f"{r['model']:<30} {r['dataset']:<10} {r['loss_function']:<10} {moca:<9} {mae:<10} {rmse:<10} {r2:<10} {ep:<5}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate CSV summary of experiment results")
    parser.add_argument("--output", "-o", default=str(SAVED_RESULTS_DIR / "results_summary.csv"),
                        help="Output CSV path (default: saved_results/results_summary.csv)")
    args = parser.parse_args()

    if not SAVED_RESULTS_DIR.exists():
        print(f"ERROR: {SAVED_RESULTS_DIR} not found. Run from project root.")
        return

    print(f"Scanning {SAVED_RESULTS_DIR}/ for results...")
    rows = collect_results(SAVED_RESULTS_DIR)

    if not rows:
        print("No results found.")
        return

    output_path = Path(args.output)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")
    print_summary(rows)


if __name__ == "__main__":
    main()
