"""T1 학교 23 · 학원 6 Feature: apt-core ↔ school_environment_v2 · v2_trade (질의일 = 계약일)."""

import numpy as np
import pandas as pd
import pytest

from apt_core.features import academy, school
import raw_record
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


def test_stale_inputs_fail(trades, schools, store_edu):
    # 갱신이 밀리면 오래된 값을 조용히 쓰지 않고 실패한다 (기준 = 학습 데이터 최대 지연)
    late = trades.iloc[[0]].assign(계약일자=pd.Timestamp("2027-07-01"))
    with pytest.raises(ValueError, match="학교 통계"):
        school.compute(late, schools)
    with pytest.raises(ValueError, match="학원 스냅샷"):
        academy.compute(trades.iloc[[0]].assign(계약일자=pd.Timestamp("2029-01-01")), store_edu, academy.clusters(store_edu))


# ---------------- R9: EDSS 원본 → 학교 표 ----------------
from conftest import EXTERNAL  # noqa: E402

EDSS_RAW = sorted(EXTERNAL.glob("0[0-9][0-9]*. *(2[1-5]).csv"))
EDSS_NAMES = {"학교속성": "school_attribute", "학교개황": "school_overview", "학급현황": "class_status", "교원현황": "teacher_status",
              "도서관운영현황": "library_status", "건물현황": "building_status", "학생변동현황": "student_movement"}


@pytest.mark.skipif(len(EDSS_RAW) < 35, reason="EDSS 원본 CSV 없음")
def test_edss_raw_to_school_table():
    merged = school.merge_edss(EDSS_RAW)
    for group, name in EDSS_NAMES.items():
        want = pd.read_parquet(EDSS / f"{name}_2021_2025.parquet")
        pd.testing.assert_frame_equal(merged[group].reset_index(drop=True), want, check_dtype=False, obj=group)
    got = school.edss_school_table(
        merged["학교속성"], merged["학교개황"], merged["학급현황"],
        pd.read_parquet(PROCESSED / "school" / "school_crosswalk_2021_2025.parquet"),
        pd.read_parquet(PROCESSED / "school_environment_2024_2026.parquet"),
        pd.read_parquet(PROCESSED / "school_code_map_v2.parquet"),
        pd.read_csv(EXTERNAL / "school_location_patch.csv", encoding="utf-8-sig"))
    want = pd.read_parquet(PROCESSED / "school" / "edss_school_environment_2021_2025.parquet")
    pd.testing.assert_frame_equal(got, want, check_dtype=False)
    assert got["학생수_보정여부"].groupby(got["개방ID"]).any().sum() == 6  # 방송통신 부설학교 6곳
    raw_record.save("edss")


# ---------------- R10: 상가정보 원본 → 교육 업종 추출 ----------------
import os  # noqa: E402

STORE_RAW = sorted((EXTERNAL / "상가정보").glob("*.csv"))


def _check_store(store_edu, paths):
    got = academy.extract_education(paths).reset_index(drop=True)
    want = store_edu[store_edu["기준월"].isin(got["기준월"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(got[want.columns], want, check_dtype=False)


@pytest.mark.skipif(not STORE_RAW, reason="상가정보 원본 없음 (약 6.5GB, gitignore)")
def test_store_extract_one_quarter(store_edu):
    _check_store(store_edu, [p for p in STORE_RAW if p.stem.endswith("202506")])


@pytest.mark.skipif(not STORE_RAW or os.environ.get("APT_CORE_SLOW") != "1", reason="25개 분기 전체 대조는 APT_CORE_SLOW=1일 때만")
def test_store_extract_all_quarters(store_edu):
    _check_store(store_edu, STORE_RAW)
    raw_record.save("store")
