"""T1 공급 20 + 재고 대비 6: apt-core ↔ supply_v2 · v2_trade."""

import os

import numpy as np
import pandas as pd
import pytest

from apt_core.features import supply
from conftest import EXTERNAL, PROCESSED


def _same(a, b, name):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{name}: 결측 위치 다름"
    ok = ~np.isnan(a)
    assert np.array_equal(a[ok], b[ok]), f"{name}: 최대 차 {np.abs(a[ok] - b[ok]).max()}"


@pytest.fixture(scope="module")
def src():
    grid = pd.read_parquet(PROCESSED / "market_leading_v2.parquet", columns=["기준월", "자치구"])
    all_gu = sorted(grid["자치구"].unique())
    return {
        "grid": grid,
        "official": supply.official_monthly(pd.read_csv(EXTERNAL / "supply_official" / "molit_seoul_permit_completion.csv", encoding="utf-8-sig")),
        "unsold": pd.read_parquet(PROCESSED / "supply" / "unsold_seoul_gu_monthly.parquet"),
        "ann": supply.prepare_announcements(pd.read_parquet(EXTERNAL / "applyhome" / "apt_announcements_seoul_20261028.parquet"), all_gu),
        "stock": pd.read_parquet(PROCESSED / "housing_stock_v2.parquet"),
    }


@pytest.fixture(scope="module")
def got(src):
    return supply.compute(src["grid"], src["official"], src["unsold"], src["ann"], src["stock"])


def test_matches_supply_v2(got, src):
    ref = pd.read_parquet(PROCESSED / "supply_v2.parquet")
    g = got.set_index(["기준월", "자치구"]).loc[pd.MultiIndex.from_frame(ref[["기준월", "자치구"]])]
    assert len(supply.COLUMNS) == 20
    for c in supply.COLUMNS:
        _same(g[c], ref[c], c)


def test_trade_rows_match_v2_trade(got, v2_trade):
    t = v2_trade[["기준월", "시군구명"]].rename(columns={"시군구명": "자치구"}).merge(got, on=["기준월", "자치구"], how="left", validate="many_to_one")
    for c in supply.COLUMNS + supply.RATIO_COLUMNS:
        if c in ("분양공고세대_3m", "분양공고세대_6m", "분양공고세대_12m", "입주예정_공고기준_24m", "미분양_구", "준공후미분양_구"):
            continue  # 구 원수치는 모델 Feature가 아님 (재고 대비로만 씀)
        _same(t[c], v2_trade[c], c)


def test_ignores_events_after_publication(src, got):
    # 04 누수 검증: 기준월 m-2월 이후 실적 · 미분양, m월 1일 이후 공고를 지우고 다시 계산해도 같아야 한다
    full = got.set_index(["기준월", "자치구"])
    for month in src["grid"]["기준월"].unique()[::7]:
        m = pd.Period(month, freq="M")
        cut = m - supply.PUBLISH_LAG
        o = src["official"][src["official"]["ref_month"] <= cut]
        u = src["unsold"][pd.PeriodIndex(src["unsold"]["ref_month"].astype(str), freq="M") <= cut]
        a = src["ann"][src["ann"]["공고일"] < m.to_timestamp()]
        sub = supply.compute(src["grid"][src["grid"]["기준월"] == month], o, u, a, src["stock"]).set_index(["기준월", "자치구"])
        pd.testing.assert_frame_equal(sub[supply.COLUMNS], full.loc[sub.index, supply.COLUMNS], check_dtype=False)


def test_shorter_lag_is_caught(src, got):
    # 시험의 시험: 공표 시차를 1개월로 줄이면 값이 달라져야 한다
    leaky = supply.compute(src["grid"], src["official"], src["unsold"], src["ann"], src["stock"], lag=1)
    assert not np.array_equal(leaky["미분양_서울"].to_numpy(float), got["미분양_서울"].to_numpy(float), equal_nan=True)


UNSOLD_RAW = sorted((EXTERNAL / "unsold_raw").glob("unsold_*.xlsx"))


def _check_unsold(src, paths):
    got = supply.parse_unsold(paths).sort_values(["ref_month", "구"]).reset_index(drop=True)
    want = src["unsold"].assign(ref_month=lambda d: d["ref_month"].astype(str))
    want = want[want["ref_month"].isin(got["ref_month"])].sort_values(["ref_month", "구"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got[want.columns], want, check_dtype=False)


@pytest.mark.skipif(not UNSOLD_RAW, reason="미분양 원본 엑셀 없음 (gitignore)")
def test_unsold_parse_sample_months(src):
    # R8: 대표 달만 (0을 '-'로 쓰던 2019년, 당월 열이 비어 다음 달 파일을 쓰는 2026-02, 최근 달)
    pick = [p for p in UNSOLD_RAW if any(m in p.name for m in ("201906", "202602", "202603", "202608"))]
    _check_unsold(src, pick)


@pytest.mark.skipif(not UNSOLD_RAW or os.environ.get("APT_CORE_SLOW") != "1", reason="전체 95개월 대조는 APT_CORE_SLOW=1일 때만 (약 10분)")
def test_unsold_parse_all_months(src):
    _check_unsold(src, UNSOLD_RAW)
