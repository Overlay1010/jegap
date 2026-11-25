"""T1 V1 기본 정제 · 기본 Feature 10개: apt-core ↔ v2_trade.parquet."""

import numpy as np
import pandas as pd
import pytest

from apt_core.features import base
from conftest import EXTERNAL, PROCESSED


@pytest.fixture(scope="module")
def cleaned():
    return base.clean(pd.read_parquet(PROCESSED / "apt_trade_raw.parquet"))


@pytest.fixture(scope="module")
def computed(cleaned):
    return base.compute(cleaned, pd.read_csv(EXTERNAL / "geocode_cache.csv", encoding="utf-8-sig"))


def test_clean_rows_and_ids_match(cleaned, v2_trade):
    t = v2_trade.sort_values("transaction_id").reset_index(drop=True)
    assert len(cleaned) == len(t) == 234_302
    assert cleaned["transaction_id"].equals(t["transaction_id"].astype(cleaned["transaction_id"].dtype))
    for c in ("단지일련번호", "시군구명", "법정동"):
        assert (cleaned[c].astype(str).str.strip().values == t[c].astype(str).values).all(), c
    assert (cleaned["계약일자"].values == pd.to_datetime(t["계약일자"]).values).all()
    assert (cleaned["거래월일자"].values == pd.to_datetime(t["거래월일자"]).values).all()


@pytest.mark.parametrize("col", base.COLUMNS)
def test_features_match_notebook(computed, v2_trade, col):
    t = v2_trade.sort_values("transaction_id").reset_index(drop=True)
    a, b = computed[col], t[col]
    if b.dtype == object:
        assert (a.fillna("<NA>").astype(str).values == b.fillna("<NA>").astype(str).values).all(), col
        return
    a, b = a.astype(float).to_numpy(), b.astype(float).to_numpy()
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{col}: 결측 위치 다름"
    ok = ~np.isnan(a)
    assert np.array_equal(a[ok], b[ok]), f"{col}: 최대 차 {np.abs(a[ok] - b[ok]).max()}"


def test_elapsed_month_is_fixed_to_base(cleaned, computed):
    # 서빙 데이터 범위가 바뀌어도 경과월 기준은 2021-07 고정 (V1은 데이터 첫 달 기준이라 범위가 바뀌면 값이 밀린다)
    later = cleaned[cleaned["계약일자"] >= "2024-01-01"]
    again = base.compute(later, pd.read_csv(EXTERNAL / "geocode_cache.csv", encoding="utf-8-sig"))
    assert again["경과월"].equals(computed.loc[later.index, "경과월"])
