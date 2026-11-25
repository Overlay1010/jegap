"""T1 시장(확정월) · 금리 Feature: apt-core ↔ v2_trade.parquet, V1 재현(시차 1), 잘라내기 검증."""

import numpy as np
import pandas as pd
import pytest

from apt_core.features import market, rate
from conftest import EXTERNAL, PROCESSED


@pytest.fixture(scope="module")
def v1_final():
    return pd.read_parquet(PROCESSED / "df_final_v1.parquet")


@pytest.fixture(scope="module")
def pool():
    from apt_core.features import base
    return market.trade_pool(base.clean(pd.read_parquet(PROCESSED / "apt_trade_raw.parquet")),
                             pd.read_parquet(PROCESSED / "apt_trade_warmup_raw.parquet"))


def _assert_same(a, b, name):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{name}: 결측 위치 다름"
    ok = ~np.isnan(a)
    assert np.array_equal(a[ok], b[ok]), f"{name}: 최대 차 {np.abs(a[ok] - b[ok]).max()}"


def test_market_matches_v2_trade(pool, v2_trade):
    feats = market.compute(pool)
    t = v2_trade[["시군구명", "거래월일자"]].assign(거래월일자=lambda d: pd.to_datetime(d["거래월일자"]))
    got = t.merge(feats, on=["시군구명", "거래월일자"], how="left", validate="many_to_one")
    for c in market.COLUMNS:
        _assert_same(got[c], v2_trade[c], c)


def test_market_reproduces_v1_with_lag1(v1_final):
    # 06 재현 검사: V1 거래만 · 시차 1이면 V1 값과 같아야 한다 (V1과 정의가 같다는 증거)
    v1 = v1_final.copy()
    v1["거래월일자"] = pd.to_datetime(v1["거래월일자"])
    v1["시군구명"] = v1["시군구명"].astype(str)
    pool_v1 = v1[market.POOL_COLS].assign(거래금액=lambda d: d["거래금액"].astype(float))
    rep = v1[["시군구명", "거래월일자"]].merge(market.compute(pool_v1, 1), on=["시군구명", "거래월일자"], how="left")
    for c in market.V1_COLUMNS:
        a, b = rep[c].to_numpy(float), v1[c].to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) < 1e-6, c


def test_market_truncation(pool):
    # 잘라내기: m월 행은 m-2월까지의 거래만 써야 한다. m-1·m월 거래를 1원짜리 자리표시로 바꿔도 값이 같아야 하고,
    # 시차 1로 계산하면 달라져야 한다 (시험의 시험). 표본 6개월
    full = {lag: market.compute(pool, lag).set_index(["시군구명", "거래월일자"]) for lag in (2, 1)}
    bad = {2: 0, 1: 0}
    for m in pd.date_range("2022-01-01", "2026-06-01", freq="10MS")[:6]:
        known = pool[pool["거래월일자"] <= m - pd.DateOffset(months=2)]
        fake = (pool[pool["거래월일자"].isin([m - pd.DateOffset(months=1), m])]
                .drop_duplicates(["시군구명", "거래월일자"]).assign(거래금액=1.0, 평당가=1.0))
        for lag in (2, 1):
            cols = [market.lag_label(c, lag) for c in market.V1_COLUMNS]
            part = market.compute(pd.concat([known, fake]), lag).set_index(["시군구명", "거래월일자"])
            a = full[lag].xs(m, level="거래월일자")[cols]
            b = part.xs(m, level="거래월일자")[cols].reindex(a.index)
            bad[lag] += int((~np.isclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)).any(axis=1).sum())
    assert bad[2] == 0
    assert bad[1] > 0


@pytest.fixture(scope="module")
def rate_monthly():
    return rate.monthly_rates(pd.read_csv(EXTERNAL / "한국은행_기준금리.csv"),
                              pd.read_csv(EXTERNAL / "mortgage_rent_loan_rate_monthly.csv"))


def test_rate_matches_v2_trade(rate_monthly, v2_trade):
    feats = rate.compute(rate_monthly)
    t = v2_trade[["거래월일자"]].assign(거래월일자=lambda d: pd.to_datetime(d["거래월일자"]))
    got = t.merge(feats, on="거래월일자", how="left", validate="many_to_one")
    cols = [c for c in feats.columns if c != "거래월일자"]
    assert len(cols) == 9
    for c in cols:
        _assert_same(got[c], v2_trade[c], c)


def test_rate_truncation(rate_monthly):
    # 공표 전 금리(기준금리 m-1월 이후, 대출금리 m-2월 이후)를 지워도 m월 값이 같아야 한다
    full = rate.compute(rate_monthly).set_index("거래월일자")
    cols = list(full.columns)
    for m in pd.date_range("2021-07-01", "2026-06-01", freq="7MS"):
        known = rate_monthly.copy()
        for col, lag in rate.RATE_LAG.items():
            known.loc[known["거래월일자"] > m - pd.DateOffset(months=lag), col] = np.nan
        part = rate.compute(known).set_index("거래월일자").loc[m, cols]
        assert np.allclose(full.loc[m, cols].to_numpy(float), part.to_numpy(float), equal_nan=True), m
