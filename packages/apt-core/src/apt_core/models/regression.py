"""11 회귀 저장 모델 추론 (LightGBM, 정답 log(거래금액)).

- 입력 열 순서 = model_config.json "features", 범주형 = "categorical"
- 범주 코드는 모델에 저장된 pandas_categorical로 맞춰진다 → 1행 질의와 전체 배치의 예측이 같다 (T4)
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model(path: str | Path) -> lgb.Booster:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return lgb.Booster(model_str=f.read())
    return lgb.Booster(model_file=str(path))


def to_matrix(rows: pd.DataFrame, config: dict) -> pd.DataFrame:
    features, cat = config["features"], set(config["categorical"])
    missing = [c for c in features if c not in rows.columns]
    if missing:
        raise KeyError(f"모델 입력 열 없음: {missing[:10]}{' …' if len(missing) > 10 else ''}")
    X = rows[features].copy()
    for c in features:
        X[c] = X[c].astype("category") if c in cat else pd.to_numeric(X[c], errors="coerce")
    return X


def predict_log(booster: lgb.Booster, rows: pd.DataFrame, config: dict, num_threads: int = 0) -> np.ndarray:
    return booster.predict(to_matrix(rows, config), num_threads=num_threads)


def predict_price(booster: lgb.Booster, rows: pd.DataFrame, config: dict, num_threads: int = 0) -> np.ndarray:
    """적정가 (만원)."""
    return np.exp(predict_log(booster, rows, config, num_threads))
