from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from .config import RunConfig
from .data import DataModule
from .metrics import evaluate_binary
from .model import build_icnn_lstm_plus, freeze_for_incremental_update


class StageProgressLogger(tf.keras.callbacks.Callback):
    def __init__(self, stage_label: str, total_epochs: int):
        super().__init__()
        self.stage_label = stage_label
        self.total_epochs = total_epochs

    def on_train_begin(self, logs=None):
        print(f"\n===== {self.stage_label} =====", flush=True)

    def on_epoch_begin(self, epoch, logs=None):
        print(f"{self.stage_label}: Epoch {epoch + 1}/{self.total_epochs}", flush=True)


def _seed_everything(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def _callbacks(patience: int, stage_label: str, total_epochs: int) -> list[tf.keras.callbacks.Callback]:
    return [
        StageProgressLogger(stage_label=stage_label, total_epochs=total_epochs),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]


def _fit_one_stage(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
    stage_label: str,
) -> None:
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=_callbacks(patience, stage_label=stage_label, total_epochs=epochs),
    )


def run_incremental_training(cfg: RunConfig) -> dict:
    run_start = time.time()
    _seed_everything(cfg.train.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    data = DataModule(cfg.data, cfg.train)
    bootstrap_df, stream_df = data.load_bootstrap_and_stream()

    # Initial phase: first 40,000 events for initial training/validation.
    bootstrap_df = bootstrap_df.head(cfg.train.initial_events).copy()
    x_init, y_init, feature_cols = data.split_xy(bootstrap_df)

    x_init_train, x_init_val, y_init_train, y_init_val = data.split_for_fit(
        x_init,
        y_init,
        ratio=cfg.train.update_train_ratio,
        seed=cfg.train.seed,
    )

    x_init_train, y_init_train = data.maybe_smote(x_init_train, y_init_train)
    x_init_train, x_init_val, scaler = data.scale_and_reshape(
        x_init_train,
        x_init_val,
        scaler=None,
        fit_scaler=True,
    )

    model = build_icnn_lstm_plus(cfg.model)

    init_stage_start = time.time()
    _fit_one_stage(
        model,
        x_init_train,
        y_init_train,
        x_init_val,
        y_init_val,
        epochs=cfg.train.initial_epochs,
        batch_size=cfg.train.fit_batch_size,
        patience=cfg.train.early_stopping_patience,
        stage_label="Initial phase",
    )
    initial_train_seconds = time.time() - init_stage_start

    init_prob_all = model.predict(x_init_val, verbose=0)
    init_prob_ransomware = init_prob_all[:, 1]
    initial_metrics = evaluate_binary(y_init_val, init_prob_ransomware, threshold=cfg.train.threshold)

    history = []

    # Incremental phase: iterate in 10,000-event windows, split each window 80:20.
    total = len(stream_df)
    total_full_batches = total // cfg.train.update_batch_events
    for i, start in enumerate(range(0, total, cfg.train.update_batch_events), start=1):
        batch_stage_start = time.time()
        end = min(start + cfg.train.update_batch_events, total)
        window = stream_df.iloc[start:end].copy()
        if len(window) < cfg.train.update_batch_events:
            break

        x_w, y_w, _ = data.split_xy(window)

        x_upd_train, x_upd_val, y_upd_train, y_upd_val = data.split_for_fit(
            x_w,
            y_w,
            ratio=cfg.train.update_train_ratio,
            seed=cfg.train.seed + i,
        )

        x_upd_train, y_upd_train = data.maybe_smote(x_upd_train, y_upd_train)
        x_upd_train, x_upd_val, scaler = data.scale_and_reshape(
            x_upd_train,
            x_upd_val,
            scaler=scaler,
            fit_scaler=False,
        )

        if cfg.train.freeze_base_on_update:
            freeze_for_incremental_update(model)

        _fit_one_stage(
            model,
            x_upd_train,
            y_upd_train,
            x_upd_val,
            y_upd_val,
            epochs=cfg.train.update_epochs,
            batch_size=cfg.train.fit_batch_size,
            patience=cfg.train.early_stopping_patience,
            stage_label=f"Incremental batch {i}/{total_full_batches}",
        )

        y_prob_all = model.predict(x_upd_val, verbose=0)
        y_prob_ransomware = y_prob_all[:, 1]
        metrics = evaluate_binary(y_upd_val, y_prob_ransomware, threshold=cfg.train.threshold)
        metrics["batch_id"] = i
        metrics["batch_start"] = int(start)
        metrics["batch_end"] = int(end)
        metrics["batch_size"] = int(len(window))
        metrics["train_pos_ratio"] = float(np.mean(y_upd_train))
        metrics["val_pos_ratio"] = float(np.mean(y_upd_val))
        metrics["train_samples"] = int(len(y_upd_train))
        metrics["val_samples"] = int(len(y_upd_val))
        metrics["batch_seconds"] = float(time.time() - batch_stage_start)
        history.append(metrics)

    model_path = cfg.output_dir / cfg.model_name
    model.save(model_path)

    results = {
        "model_path": str(model_path),
        "num_features": len(feature_cols),
        "num_incremental_batches": len(history),
        "initial_train_seconds": float(initial_train_seconds),
        "initial_metrics": initial_metrics,
        "total_runtime_seconds": float(time.time() - run_start),
        "history": history,
    }

    result_path = cfg.output_dir / "incremental_metrics.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train iCNN-LSTM+ with batch incremental updates")
    parser.add_argument("--train-csv", type=str, default="SILRAD-dataset/fasttext-trainmodel.csv")
    parser.add_argument("--stream-csv", type=str, default="SILRAD-dataset/fasttext-testmodel.csv")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--initial-events", type=int, default=40000)
    parser.add_argument("--update-events", type=int, default=10000)
    parser.add_argument("--initial-epochs", type=int, default=12)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--fit-batch-size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig()
    cfg.data.train_csv = Path(args.train_csv)
    cfg.data.stream_csv = Path(args.stream_csv)
    cfg.output_dir = Path(args.output_dir)
    cfg.train.initial_events = args.initial_events
    cfg.train.update_batch_events = args.update_events
    cfg.train.initial_epochs = args.initial_epochs
    cfg.train.update_epochs = args.update_epochs
    cfg.train.fit_batch_size = args.fit_batch_size
    cfg.train.threshold = args.threshold

    results = run_incremental_training(cfg)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
