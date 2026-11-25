"""일치 테스트 공용: 기준값 = 커밋된 노트북 산출물."""

from pathlib import Path

import pandas as pd
import pytest


def _root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / "data" / "v2_5year_seoul" / "processed").is_dir():
            return p
    raise FileNotFoundError("프로젝트 루트(data/v2_5year_seoul/processed)를 찾을 수 없습니다")


ROOT = _root()
PROCESSED = ROOT / "data" / "v2_5year_seoul" / "processed"
EXTERNAL = ROOT / "data" / "external"
MODEL_DIR = ROOT / "models" / "regression" / "v2"


@pytest.fixture(scope="session")
def v2_trade() -> pd.DataFrame:
    t = pd.read_parquet(PROCESSED / "v2_trade.parquet")
    t["단지일련번호"] = t["단지일련번호"].astype(str)
    return t


@pytest.fixture(scope="session")
def registry() -> pd.DataFrame:
    from apt_core import registry as reg
    return reg.load(PROCESSED / "v2_feature_registry.csv")


@pytest.fixture(scope="session")
def model_config() -> dict:
    from apt_core.models import regression
    return regression.load_config(MODEL_DIR / "model_config.json")


@pytest.fixture(scope="session")
def trades_raw() -> pd.DataFrame:
    return pd.concat([pd.read_parquet(PROCESSED / "apt_trade_warmup_raw.parquet"),
                      pd.read_parquet(PROCESSED / "apt_trade_raw.parquet")], ignore_index=True)


@pytest.fixture(scope="session")
def weekly_index() -> pd.DataFrame:
    return pd.read_csv(EXTERNAL / "rone_weekly" / "apt_sale_index_weekly_gu.csv", dtype={"주": str})
