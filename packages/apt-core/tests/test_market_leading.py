"""T1 시장 선행 16개: apt-core(R-ONE · 한국은행 원본부터) ↔ market_leading_v2 · v2_trade."""

import numpy as np
import pandas as pd
import pytest

from apt_core.features import market_leading as ml
from conftest import EXTERNAL, PROCESSED


def _same(a, b, name):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{name}: 결측 위치 다름"
    ok = ~np.isnan(a)
    assert np.array_equal(a[ok], b[ok]), f"{name}: 최대 차 {np.abs(a[ok] - b[ok]).max()}"


@pytest.fixture(scope="module")
def parts():
    xl = lambda name: pd.read_excel(EXTERNAL / name, sheet_name=0, header=None)  # noqa: E731
    return {
        "sale_supply": ml.clean_supply(ml.read_rone_csv(EXTERNAL / "(월) 매매수급동향_아파트.csv"), "매매수급지수"),
        "jeonse_supply": ml.clean_supply(ml.read_rone_csv(EXTERNAL / "(월) 전세수급동향_아파트.csv"), "전세수급지수"),
        "sale_price": ml.clean_price(ml.read_rone_csv(EXTERNAL / "(월) 매매가격지수_주택종합.csv"), "매매"),
        "jeonse_price": ml.clean_price(ml.read_rone_csv(EXTERNAL / "(월) 전세가격지수_주택종합.csv"), "전세"),
        "housing_csi": ml.clean_bok(xl("주택가격전망CSI와 미분양주택.xlsx"), "주택가격전망CSI"),
        "ccsi": ml.clean_bok(xl("소비자심리지수(CCSI).xlsx"), "CCSI"),
    }


@pytest.fixture(scope="module")
def base(parts):
    # V2는 2026-06까지 정제했다. 원본 파일에 그 뒤 달이 있어도 V2 값과 비교하려고 같은 범위로 자른다
    b = ml.base_table(**parts)
    return b[b["기준월"] <= pd.Period("2026-06", freq="M")].reset_index(drop=True)


@pytest.fixture(scope="module")
def got(base):
    return ml.compute(base)


def test_matches_market_leading_v2(got):
    ref = pd.read_parquet(PROCESSED / "market_leading_v2.parquet")
    g = got.assign(기준월=got["기준월"].astype(str)).set_index(["기준월", "자치구"]).loc[pd.MultiIndex.from_frame(ref[["기준월", "자치구"]])]
    assert len(ml.COLUMNS) == 16
    assert (g["권역"].to_numpy() == ref["권역"].to_numpy()).all()
    for c in ml.COLUMNS:
        _same(g[c], ref[c], c)


def test_matches_v2_trade(got, v2_trade):
    t = v2_trade[["기준월", "시군구명"]].merge(got.assign(기준월=got["기준월"].astype(str)).rename(columns={"자치구": "시군구명"}),
                                             on=["기준월", "시군구명"], how="left", validate="many_to_one")
    for c in ml.COLUMNS:
        _same(t[c], v2_trade[c], c)


def test_ignores_unpublished_months(base, got):
    # 기준월 m 행은 부동산원 m-2월 · 한국은행 m-1월까지만 써야 한다: 그 뒤 값을 지워도 같아야 한다
    full = got.set_index(["기준월", "자치구"])
    for m in pd.period_range("2021-09", "2026-06", freq="M")[::6]:
        cut = base[base["기준월"] <= m].copy()
        for col, lag in ml.PUBLICATION_LAG.items():
            cut.loc[cut["기준월"] > m - lag, col] = np.nan
        sub = ml.compute(cut).set_index(["기준월", "자치구"]).xs(m, level="기준월", drop_level=False)
        pd.testing.assert_frame_equal(sub[ml.COLUMNS], full.loc[sub.index, ml.COLUMNS])


def test_shorter_lag_is_caught(base, got):
    leaky = ml.compute(base, {k: v - 1 for k, v in ml.PUBLICATION_LAG.items()})
    assert not np.array_equal(leaky["매매수급지수_lag2"].to_numpy(float), got["매매수급지수_lag2"].to_numpy(float), equal_nan=True)


def test_missing_month_fails(base):
    with pytest.raises(ValueError, match="월이 빠지거나"):
        ml.compute(base[base["기준월"] != pd.Period("2024-03", freq="M")])
