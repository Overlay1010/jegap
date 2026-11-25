"""T1 단지 Feature 21개: apt-core ↔ 05b 중간 산출물 · v2_trade.parquet.

공시가격 호 단위 원본은 gitignore(약 63MB)라 CI에는 없다 → 공시 관련 테스트는 원본이 있을 때만 (설계 1-4 'CI로 못 하는 부분')."""

import numpy as np
import pandas as pd
import pytest

from apt_core.features import base, complex as cx
from conftest import EXTERNAL, PROCESSED

OFFICIAL_RAW = sorted((EXTERNAL / "official_price").glob("apt_official_price_seoul_*.parquet"))
needs_official = pytest.mark.skipif(not OFFICIAL_RAW, reason="공시가격 호 단위 원본 없음 (gitignore)")


def _same(a, b, name):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{name}: 결측 위치 다름"
    ok = ~np.isnan(a)
    assert np.array_equal(a[ok], b[ok]), f"{name}: 최대 차 {np.abs(a[ok] - b[ok]).max()}"


@pytest.fixture(scope="module")
def trades():
    t = base.clean(pd.read_parquet(PROCESSED / "apt_trade_raw.parquet"))
    return t.assign(lot=cx.lot_of(t))


@pytest.fixture(scope="module")
def complexes(trades):
    return cx.complexes_of(trades)


@pytest.fixture(scope="module")
def attrs(complexes):
    return cx.attributes(complexes, pd.read_parquet(EXTERNAL / "building_register_apt_seoul_20261022.parquet"),
                         pd.read_parquet(EXTERNAL / "building_register_recap_seoul_20261022.parquet"))


@pytest.fixture(scope="module")
def zone_of(complexes):
    return cx.rebuild_zones(cx.read_jeongbi(EXTERNAL / "jeongbi" / "seoul_jeongbi_status_202606.xlsx"), complexes)


@pytest.fixture(scope="module")
def units():
    return pd.concat([cx.official_units(pd.read_parquet(p)) for p in OFFICIAL_RAW], ignore_index=True)


def test_attributes_match_notebook(attrs):
    ref = pd.read_parquet(PROCESSED / "complex_attributes_v2.parquet")
    assert attrs["단지일련번호"].tolist() == ref["단지일련번호"].tolist() and attrs["lot"].tolist() == ref["lot"].tolist()
    for c in ["사용승인연도"] + cx.ATTR_COLUMNS:
        _same(attrs[c], ref[c], c)


def test_rebuild_zones_match_notebook(zone_of):
    ref = pd.read_parquet(PROCESSED / "complex_rebuild_zone_v2.parquet")
    pd.testing.assert_frame_equal(zone_of, ref.reset_index(drop=True), check_dtype=False)


def test_stage_features_match_v2_trade(trades, zone_of, v2_trade):
    t = v2_trade.sort_values("transaction_id").reset_index(drop=True)
    got = cx.stage_features(trades, zone_of)
    for c in cx.STAGE_COLUMNS:
        assert (got[c].to_numpy() == t[c].to_numpy()).all(), c


def test_stage_ignores_future_notices(trades, zone_of):
    # 계약일 이후 고시일을 지운 구역표로 다시 계산해도 같아야 한다 (05b 누수 검증)
    rng = np.random.default_rng(3)
    for d in pd.to_datetime(rng.choice(trades["계약일자"].unique(), 50, replace=False)):
        tr = trades[trades["계약일자"].eq(d)]
        z = zone_of.copy()
        for col, _ in cx.STAGES:
            z.loc[z[col] >= d, col] = pd.NaT
        assert cx.stage_features(tr, z).equals(cx.stage_features(tr, zone_of)), d


def test_release_month_boundary():
    edge = pd.Series(pd.to_datetime(["2023-06-30", "2023-07-01"]))
    assert list(cx.available_year(edge)) == [2022, 2023]


@needs_official
def test_official_tables_and_mix_match_notebook(units, complexes):
    price_lot, _ = cx.official_tables(units)
    ref = pd.read_parquet(PROCESSED / "complex_official_price_v2.parquet")
    got = complexes[["단지일련번호", "lot"]].merge(price_lot, on="lot", how="left").dropna(subset=["기준연도"]).astype({"기준연도": int})
    pd.testing.assert_frame_equal(got.reset_index(drop=True), ref.reset_index(drop=True), check_dtype=False)
    mix = cx.unit_mix(units, complexes)
    ref_mix = pd.read_parquet(PROCESSED / "complex_unit_mix_v2.parquet")
    for c in ["면적기준연도"] + cx.MIX_COLUMNS:
        _same(mix[c], ref_mix[c], c)


@needs_official
def test_all_21_columns_match_v2_trade(trades, complexes, attrs, zone_of, units, v2_trade):
    price_lot, price_band = cx.official_tables(units)
    got = cx.compute(trades, price_lot, price_band, zone_of, attrs, cx.unit_mix(units, complexes))
    t = v2_trade.sort_values("transaction_id").reset_index(drop=True)
    assert len(cx.COLUMNS) == 21
    for c in cx.COLUMNS:
        _same(got[c], t[c], c)


@needs_official
def test_official_ignores_unreleased_years(trades, units):
    price_lot, price_band = cx.official_tables(units)
    rng = np.random.default_rng(3)
    for d in pd.to_datetime(rng.choice(trades["계약일자"].unique(), 30, replace=False)):
        tr = trades[trades["계약일자"].eq(d)]
        known = d.year if d.month >= cx.RELEASE_MONTH else d.year - 1
        a = cx.official_features(tr, price_lot[price_lot["기준연도"] <= known], price_band[price_band["기준연도"] <= known])
        assert a.equals(cx.official_features(tr, price_lot, price_band)), d
