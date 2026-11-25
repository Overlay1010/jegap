"""T4 추론: apt-core 예측 ↔ 노트북 방식 예측(저장 모델에 범주형만 바꿔 넣기), 1행 질의 ↔ 배치."""

import numpy as np
import pytest

from apt_core.models import regression
from conftest import MODEL_DIR


@pytest.fixture(scope="module")
def booster():
    return regression.load_model(MODEL_DIR / "lgbm_regression_v2.txt.gz")


@pytest.fixture(scope="module")
def sample(v2_trade):
    return v2_trade[v2_trade["데이터구분"].eq("test")].sample(2_000, random_state=0)


def test_matches_notebook_way(booster, sample, model_config):
    # 11 노트북 방식: v2_trade[FEATURES]에서 object 열만 category로 바꿔 predict
    X = sample[model_config["features"]].copy()
    for c in model_config["categorical"]:
        X[c] = X[c].astype("category")
    expected = booster.predict(X)
    got = regression.predict_log(booster, sample, model_config)
    assert np.array_equal(got, expected)


def test_single_row_equals_batch(booster, sample, model_config):
    batch = regression.predict_log(booster, sample, model_config)
    for i in range(0, len(sample), 100):
        one = regression.predict_log(booster, sample.iloc[[i]], model_config)
        assert one[0] == batch[i], (i, one[0], batch[i])


def test_missing_feature_raises(booster, sample, model_config):
    with pytest.raises(KeyError):
        regression.predict_log(booster, sample.drop(columns=["층"]), model_config)
