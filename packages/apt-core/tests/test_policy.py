"""T1 정책 41개: apt-core ↔ policy_v2 · policy_trade_v2 · v2_trade. 정책 CSV는 현재 파일(10 ~ 11월 추가분 포함) 그대로 —
2026-06 이후 발표분이 과거 Feature를 바꾸지 않는다는 것도 함께 확인된다."""

import json

import numpy as np
import pandas as pd
import pytest

from apt_core.features import base, policy
from conftest import EXTERNAL, PROCESSED


def _same(a, b, name):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{name}: 결측 위치 다름"
    ok = ~np.isnan(a)
    assert np.array_equal(a[ok], b[ok]), f"{name}: 최대 차 {np.abs(a[ok] - b[ok]).max()}"


@pytest.fixture(scope="module")
def inputs():
    return {"events": policy.read_events(EXTERNAL / "housing_policy_events_v2.csv"),
            "zone_gu": policy.read_zones(EXTERNAL / "policy_zone_status_by_gu_v2.csv"),
            "zone_dong": policy.read_zones(EXTERNAL / "policy_zone_status_by_dong_v2.csv")}


@pytest.fixture(scope="module")
def trades():
    t = base.clean(pd.read_parquet(PROCESSED / "apt_trade_raw.parquet"))
    f = base.compute(t, pd.read_csv(EXTERNAL / "geocode_cache.csv", encoding="utf-8-sig"))
    t["거래유형"], t["위도"], t["경도"] = f["거래유형"], f["위도"], f["경도"]
    t["법정동"] = t["법정동"].astype(str).str.strip()
    t["평당가"] = t["거래금액"].astype(float) / (t["전용면적"].astype(float) / 3.3058)
    return t


@pytest.fixture(scope="module")
def apt_dong(inputs, trades):
    xy = trades.drop_duplicates("단지일련번호")[["단지일련번호", "단지명", "법정동", "위도", "경도"]]
    geo = json.loads((EXTERNAL / "jeongbi" / "seongsu_strategic_zone.geojson").read_text(encoding="utf-8"))
    rules = pd.read_csv(EXTERNAL / "policy_permit_complex_rules.csv", encoding="utf-8-sig")
    return policy.permit_rules(rules, policy.apartment_permit_zones(inputs["zone_dong"]), xy, geo)


@pytest.fixture(scope="module")
def grid():
    return pd.read_parquet(PROCESSED / "market_leading_v2.parquet", columns=["기준월", "자치구", "권역"])


@pytest.fixture(scope="module")
def region(inputs, grid, trades):
    return policy.region_month(inputs["events"], grid, inputs["zone_gu"], policy.reference_trades(trades))


@pytest.fixture(scope="module")
def expected(trades):
    return policy.expected_price(trades)


def test_region_month_matches_notebook(region):
    ref = pd.read_parquet(PROCESSED / "policy_v2.parquet")
    got = region.set_index(["기준월", "자치구"]).loc[pd.MultiIndex.from_frame(ref[["기준월", "자치구"]])]
    assert len(policy.REGION_FEATURES) == 32
    for c in policy.REGION_FEATURES:
        _same(got[c], ref[c], c)


def test_expected_price_matches_notebook(expected, v2_trade):
    ref = pd.read_parquet(PROCESSED / "policy_trade_v2.parquet").sort_values("transaction_id").reset_index(drop=True)
    _same(expected["예상가격_만원"], ref["예상가격_만원"], "예상가격_만원")
    assert (expected["예상가격_출처"].to_numpy() == ref["예상가격_출처"].to_numpy()).all()
    _same(expected["예상가격_단지만"], v2_trade.sort_values("transaction_id")["예상가격_단지만"], "예상가격_단지만")


def test_trade_features_match_notebook(inputs, apt_dong, trades, expected):
    got = policy.trade_features(trades.assign(예상가격_만원=expected["예상가격_만원"]), inputs["events"], inputs["zone_gu"], apt_dong)
    ref = pd.read_parquet(PROCESSED / "policy_trade_v2.parquet").sort_values("transaction_id").reset_index(drop=True)
    for c in policy.TRADE_FEATURES:
        _same(got[c], ref[c], c)
    assert (got["토허_근거"].fillna("").to_numpy() == ref["토허_근거"].fillna("").to_numpy()).all()


def _known(events, cutoff):
    return events[events["announce_date"].fillna(events["effective_date"]) < cutoff]


def test_region_month_ignores_later_announcements(inputs, grid, trades, region):
    # 03 누수 검증: 기준월 이후 발표 정책을 지우고 다시 계산해도 같아야 한다 (표본 12개월)
    ref = policy.reference_trades(trades)
    for month in grid["기준월"].unique()[::5]:
        known = _known(inputs["events"], pd.Period(month, freq="M").to_timestamp())
        sub = policy.region_month(known, grid[grid["기준월"] == month], inputs["zone_gu"], ref, known_ids=set(known["event_id"]))
        full = region[region["기준월"] == month]
        diff = sub.set_index("자치구")[policy.REGION_FEATURES].compare(full.set_index("자치구")[policy.REGION_FEATURES])
        assert diff.empty, (month, diff)


def test_trade_ignores_later_announcements(inputs, apt_dong, trades, expected):
    t = trades.assign(예상가격_만원=expected["예상가격_만원"])
    for d in pd.Series(t["계약일자"].unique()).sample(40, random_state=7):
        tr = t[t["계약일자"] == d]
        sub = policy.trade_features(tr, inputs["events"], inputs["zone_gu"], apt_dong, known_ids=set(_known(inputs["events"], d)["event_id"]))
        full = policy.trade_features(tr, inputs["events"], inputs["zone_gu"], apt_dong)
        pd.testing.assert_frame_equal(sub[policy.TRADE_FEATURES], full[policy.TRADE_FEATURES])


def test_leak_check_catches_injected_leak(inputs, grid):
    # 시험의 시험: 10.15 지정(P156)을 발표 전 날짜로 잘못 넣으면 2025-10 기준월에서 21개 구가 달라져야 한다
    bad = inputs["zone_gu"].copy()
    bad.loc[bad["start_event_id"].eq("P156"), ["start_date", "start_announce_date"]] = pd.Timestamp("2025-09-25")
    probe = grid[grid["기준월"] == "2025-10"].assign(cutoff=pd.Timestamp("2025-10-01"))
    known = set(_known(inputs["events"], "2025-10-01")["event_id"])
    diff = policy.zone_flags(probe, "자치구", "cutoff", bad).ne(policy.zone_flags(probe, "자치구", "cutoff", bad, known)).sum()
    assert diff["투기과열지구"] == 21


def test_rule_event_names_are_checked(tmp_path):
    ev = pd.read_csv(EXTERNAL / "housing_policy_events_v2.csv", encoding="utf-8-sig")
    ev.loc[ev["event_id"].eq("P034"), "policy_name"] = "바뀐 이름"
    ev.to_csv(tmp_path / "e.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError):
        policy.read_events(tmp_path / "e.csv")
