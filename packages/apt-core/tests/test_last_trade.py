"""T1 직전 거래 · 시세 보정: apt-core 계산 ↔ v2_trade.parquet (질의일 = 계약일)."""

import numpy as np
import pandas as pd
import pytest

from apt_core.features import last_trade


@pytest.fixture(scope="module")
def computed(v2_trade, trades_raw, weekly_index):
    ref = last_trade.build_reference(trades_raw)
    q = v2_trade[["단지일련번호", "전용면적", "시군구명", "계약일자"]]
    return last_trade.compute(q, ref, weekly_index, date_col="계약일자")


@pytest.mark.parametrize("col", last_trade.COLUMNS)
def test_matches_notebook_exactly(computed, v2_trade, col):
    a, b = computed[col].to_numpy(float), v2_trade[col].to_numpy(float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{col}: 결측 위치 다름"
    ok = ~np.isnan(a)
    assert np.array_equal(a[ok], b[ok]), f"{col}: 최대 차 {np.abs(a[ok] - b[ok]).max()}"


def test_point_in_time_visible_only_after_report_lag(computed):
    # 06 시점 검증: 직전 거래는 모두 질의일 30일 전 이전
    assert (computed["직전거래_경과일"].dropna() >= last_trade.REPORT_DAYS).all()


def test_shorter_lag_must_fail(v2_trade, trades_raw, weekly_index, monkeypatch):
    # 시험의 시험: 신고 기한을 29일로 줄이면 결과가 달라져야 한다 (누출 검사가 실제로 동작)
    sample = v2_trade.sample(5_000, random_state=0)
    q = sample[["단지일련번호", "전용면적", "시군구명", "계약일자"]]
    monkeypatch.setattr(last_trade, "REPORT_DAYS", 29)
    leaky = last_trade.compute(q, last_trade.build_reference(trades_raw), weekly_index)
    assert not np.array_equal(leaky["직전거래가_동일면적"].to_numpy(float), sample["직전거래가_동일면적"].to_numpy(float), equal_nan=True)


def test_query_date_path_matches_rows(v2_trade, trades_raw, weekly_index):
    # 질의 경로 = 거래 경로: 같은 (단지 · 면적 · 계약일) 1행 질의가 학습 행과 같다
    ref = last_trade.build_reference(trades_raw)
    rows = v2_trade[v2_trade["데이터구분"].eq("test")].sample(50, random_state=1)
    for _, r in rows.iterrows():
        one = last_trade.compute(r.to_frame().T[["단지일련번호", "전용면적", "시군구명", "계약일자"]], ref, weekly_index)
        for c in last_trade.COLUMNS:
            a, b = float(one[c].iloc[0]), float(r[c])
            assert (np.isnan(a) and np.isnan(b)) or a == b, (r["transaction_id"], c, a, b)


def test_stale_weekly_index_fails(v2_trade, trades_raw, weekly_index):
    q = v2_trade.iloc[[0]][["단지일련번호", "전용면적", "시군구명"]].assign(계약일자=pd.Timestamp("2027-03-01"))
    with pytest.raises(ValueError, match="주간 지수"):
        last_trade.compute(q, last_trade.build_reference(trades_raw), weekly_index)
