from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import DataConfig, TrainingConfig


class DataModule:
    def __init__(self, data_cfg: DataConfig, train_cfg: TrainingConfig):
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg

    def _read_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        # Handle BOM-prefixed first column in this dataset.
        if "\ufeffevent.code" in df.columns:
            df = df.rename(columns={"\ufeffevent.code": "event.code"})
        return df

    def load_bootstrap_and_stream(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        bootstrap_df = self._read_csv(self.data_cfg.train_csv)
        stream_df = self._read_csv(self.data_cfg.stream_csv)
        return bootstrap_df, stream_df

    def split_xy(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        feature_cols = [c for c in df.columns if c != self.data_cfg.label_col]
        x = df[feature_cols].to_numpy(dtype=np.float32)
        y = df[self.data_cfg.label_col].to_numpy(dtype=np.int32)
        return x, y, feature_cols

    def scale_and_reshape(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
        scaler = scaler or StandardScaler()

        if fit_scaler:
            x_train = scaler.fit_transform(x_train)
        else:
            x_train = scaler.transform(x_train)

        x_test = scaler.transform(x_test)

        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        return x_train, x_test, scaler

    def maybe_smote(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.train_cfg.use_smote:
            return x, y

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return x, y

        # SMOTE requires enough minority examples for nearest-neighbor synthesis.
        if counts.min() < 2:
            return x, y

        smote = SMOTE(random_state=self.train_cfg.seed)
        x_resampled, y_resampled = smote.fit_resample(x, y)
        return x_resampled.astype(np.float32), y_resampled.astype(np.int32)

    def split_for_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ratio: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        classes = np.unique(y)
        stratify = y if len(classes) > 1 else None
        return train_test_split(
            x,
            y,
            train_size=ratio,
            random_state=seed,
            stratify=stratify,
        )
