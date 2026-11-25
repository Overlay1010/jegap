"""T1 인구 15개 + 재고: apt-core(KOSIS 원본부터) ↔ demographics_v2 · housing_stock_v2 · v2_trade."""

import numpy as np
import pandas as pd
import pytest

from apt_core.features import demographics as demo
from conftest import EXTERNAL, PROCESSED


def _same(a, b, name):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{name}: 결측 위치 다름"
    ok = ~np.isnan(a)
    assert np.array_equal(a[ok], b[ok]), f"{name}: 최대 차 {np.abs(a[ok] - b[ok]).max()}"


@pytest.fixture(scope="module")
def raw():
    return {p.stem: pd.read_parquet(p) for p in sorted((EXTERNAL / "kosis").glob("*.parquet"))}


@pytest.fixture(scope="module")
def months():
    g = pd.read_parquet(PROCESSED / "market_leading_v2.parquet", columns=["기준월"])
    return pd.PeriodIndex(g["기준월"].unique(), freq="M")


@pytest.fixture(scope="module")
def base(raw):
    return demo.base_table(raw)


@pytest.fixture(scope="module")
def got(base, months):
    return demo.compute(base, months).assign(기준월=lambda d: d["month"].astype(str))


def test_matches_demographics_v2(got):
    ref = pd.read_parquet(PROCESSED / "demographics_v2.parquet")
    g = got.set_index(["기준월", "자치구"]).loc[pd.MultiIndex.from_frame(ref[["기준월", "자치구"]])]
    assert len(demo.COLUMNS) == 15
    for c in demo.COLUMNS:
        _same(g[c], ref[c], c)


def test_matches_v2_trade(got, v2_trade):
    t = v2_trade[["기준월", "시군구명"]].merge(got.rename(columns={"자치구": "시군구명"}), on=["기준월", "시군구명"],
                                             how="left", validate="many_to_one")
    for c in demo.COLUMNS:
        _same(t[c], v2_trade[c], c)


def test_housing_stock_matches(raw, got, months):
    gus = sorted(got["자치구"].unique())
    st = demo.housing_stock(raw["DT_1JU1501"], gus, months)
    lv = got[["자치구", "month", "기준월"] + demo.LEVEL_COLUMNS].merge(st, on=["자치구", "month"])
    ref = pd.read_parquet(PROCESSED / "housing_stock_v2.parquet")
    g = lv.set_index(["기준월", "자치구"]).loc[pd.MultiIndex.from_frame(ref[["기준월", "자치구"]])]
    for c in demo.LEVEL_COLUMNS + ["조사연도", "주택재고", "아파트재고"]:
        _same(g[c], ref[c], c)


def test_ignores_unpublished_months(base, got, months):
    full = got.set_index(["month", "자치구"])
    for m in months[::7]:
        cut = base[base.index.get_level_values("data_month") <= m - demo.PUBLISH_LAG]
        sub = demo.compute(cut, pd.PeriodIndex([m])).set_index(["month", "자치구"])
        pd.testing.assert_frame_equal(sub[demo.COLUMNS], full.loc[sub.index, demo.COLUMNS])


def test_shorter_lag_is_caught(base, got, months):
    leaky = demo.compute(base, months, lag=1)
    assert not np.array_equal(leaky["순이동률_3m"].to_numpy(float), got["순이동률_3m"].to_numpy(float))
