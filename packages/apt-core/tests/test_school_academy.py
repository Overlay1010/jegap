"""T1 학교 23 · 학원 6 Feature: apt-core ↔ school_environment_v2 · v2_trade (질의일 = 계약일)."""

import numpy as np
import pandas as pd
import pytest

from apt_core.features import academy, school
from conftest import PROCESSED

EDSS = PROCESSED / "edss"


def _same(a, b, name):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{name}: 결측 위치 다름"
    ok = ~np.isnan(a)
    assert np.array_equal(a[ok], b[ok]), f"{name}: 최대 차 {np.abs(a[ok] - b[ok]).max()}"


@pytest.fixture(scope="module")
def trades(v2_trade):
    return v2_trade.sort_values("transaction_id").reset_index(drop=True)


@pytest.fixture(scope="module")
def ref(trades):
    r = pd.read_parquet(PROCESSED / "school_environment_v2.parquet")
    return trades[["transaction_id"]].merge(r, on="transaction_id", how="left", validate="one_to_one")


@pytest.fixture(scope="module")
def schools():
    return school.school_table(
        pd.read_parquet(PROCESSED / "school" / "edss_school_environment_2021_2025.parquet"),
        pd.read_parquet(EDSS / "student_movement_2021_2025.parquet"),
        pd.read_parquet(EDSS / "school_overview_2021_2025.parquet"),
        pd.read_parquet(EDSS / "building_status_2021_2025.parquet"))


@pytest.fixture(scope="module")
def store_edu():
    return pd.read_parquet(PROCESSED / "academy" / "store_education_2020_2026.parquet")


def test_school_matches_notebook(trades, schools, ref):
    got = school.compute(trades, schools)
    assert len(school.COLUMNS) == 23
    assert (got["학교기준연도"].to_numpy() == ref["학교기준연도"].to_numpy()).all()
    for c in school.DISPLAY:
        assert (got[c].to_numpy() == ref[c].to_numpy()).all(), c
    for c in school.COLUMNS:
        _same(got[c], trades[c], c)


def test_school_year_published_before_deal():
    d = pd.Series(pd.to_datetime(["2024-05-31", "2024-06-01", "2027-03-01"]))
    assert list(school.school_year(d, 2021, 2025)) == [2023, 2024, 2025]


def test_academy_clusters_match_notebook(store_edu):
    got = academy.clusters(store_edu)
    want = pd.read_parquet(PROCESSED / "academy" / "academy_clusters_202012.parquet")
    pd.testing.assert_frame_equal(got, want, check_dtype=False)
    assert got.loc[got["3대학원가"], "자치구"].tolist() == ["강남구", "노원구", "양천구"]


def test_academy_matches_notebook(trades, store_edu, ref):
    got = academy.compute(trades, store_edu, academy.clusters(store_edu))
    for c in ["학원스냅샷기준월", "학원스냅샷_경과개월"] + academy.DISPLAY:
        assert (got[c].astype(str).to_numpy() == ref[c].astype(str).to_numpy()).all(), c
    for c in academy.COLUMNS:
        _same(got[c], trades[c], c)


def test_academy_snapshot_not_before_release(store_edu):
    snaps = academy.normal_snapshots(store_edu)
    assert "202303" not in snaps  # 결손 분기는 건너뜀
    s = academy.snapshot_for(pd.Series([202103, 202104]), snaps)
    assert (s["학원스냅샷_경과개월"] >= academy.RELEASE_LAG_MONTHS).all()
