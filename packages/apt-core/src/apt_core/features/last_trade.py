"""같은 단지 · 같은 면적 직전 거래 + 시세 보정 (06_feature_merge 셀 22 · 24 이식, 레지스트리 source = 06_last_trade).

시점 규칙:
- 직전 거래 = 질의일 30일 전까지 계약된 같은 단지 · 면적(전용면적 반올림) 중개거래 (해제 · 직거래 제외)
  · 실거래 신고 기한 30일 → 그때까지는 공개됐다고 본다. 같은 공개일에 여러 건이면 마지막 행이 모두 반영
- 시세 보정 = 부동산원 주간 구 지수. 지금 지수 = 질의일 전에 공표된 마지막 주(월요일 기준 주 + 3일 목요일 공표),
  직전 거래 시점 지수 = 직전 계약일이 속한 주. 직전 계약일이 지수 시작 주 전이면 비운다

질의일은 학습 행이면 계약일, 서빙이면 기준일(매달 1일 · 매주 금요일, v4-design 2-1)이다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REPORT_DAYS = 30
INDEX_LAG = pd.Timedelta(days=3)  # 월요일 기준 주 → 목요일 공표
MAX_INDEX_STALE_DAYS = 14  # 학습 데이터 최대 (명절로 한 주 빠진 경우). 넘으면 주간 지수 수집이 멈춘 것

COLUMNS = ["직전거래가_동일면적", "직전3건중위_동일면적", "직전거래_경과일", "직전거래_층", "동일면적_누적거래수",
           "직전거래후_구지수변화", "직전거래가_시세보정"]


def build_reference(trades_raw: pd.DataFrame) -> pd.DataFrame:
    """실거래 원본(국토부 API 열) → 공개일 기준 참고 표. 06 셀 22의 ref 계산 그대로."""
    ref = trades_raw[trades_raw["해제여부"].ne("O") & trades_raw["거래유형"].ne("직거래")].copy()
    ref["계약일자"] = pd.to_datetime(dict(year=ref["계약년도"], month=ref["계약월"], day=ref["계약일"]))
    ref["단지일련번호"] = ref["단지일련번호"].astype(str)
    ref["면적키"] = ref["전용면적"].astype(float).round(0)
    ref["거래금액"] = pd.to_numeric(ref["거래금액"].astype(str).str.replace(",", ""), errors="coerce")
    ref = ref.sort_values("계약일자", kind="stable")
    by_unit = ref.groupby(["단지일련번호", "면적키"])
    ref["직전3건중위"] = by_unit["거래금액"].transform(lambda s: s.rolling(3, min_periods=1).median())
    ref["누적거래수"] = by_unit.cumcount() + 1
    ref["공개일"] = ref["계약일자"] + pd.Timedelta(days=REPORT_DAYS)
    ref = ref.drop_duplicates(["단지일련번호", "면적키", "공개일"], keep="last")
    return (ref[["단지일련번호", "면적키", "공개일", "계약일자", "거래금액", "층", "직전3건중위", "누적거래수"]]
            .rename(columns={"계약일자": "직전계약일"}).sort_values("공개일"))


def prepare_index(weekly_index: pd.DataFrame) -> pd.DataFrame:
    """부동산원 주간 구 매매지수 (자치구, 주, 지수). 주는 'YYYYMMDD' 문자열 또는 날짜."""
    wk = weekly_index[["주", "자치구", "지수"]].copy()
    wk["주"] = pd.to_datetime(wk["주"].astype(str))
    return wk.sort_values("주")


def _gu_index(wk: pd.DataFrame, dates: pd.Series, gu: pd.Series, published_before: bool) -> np.ndarray:
    # published_before=True: (주 + 3일) < 날짜인 마지막 주 / False: 주 ≤ 날짜인 마지막 주 (그 날짜가 속한 주)
    q = pd.DataFrame({"pos": np.arange(len(dates)), "key": dates.to_numpy(), "자치구": gu.to_numpy()}).dropna(subset=["key"])
    ref = wk.assign(key=wk["주"] + (INDEX_LAG if published_before else pd.Timedelta(0)))
    m = pd.merge_asof(q.sort_values("key"), ref.sort_values("key"), on="key", by="자치구",
                      direction="backward", allow_exact_matches=not published_before)
    value = np.full(len(dates), np.nan)
    value[m["pos"].to_numpy()] = m["지수"].to_numpy()
    if published_before and len(m):
        stale = (m["key"] - (m["주"] + INDEX_LAG)).dt.days
        if (worst := stale.max()) > MAX_INDEX_STALE_DAYS:
            raise ValueError(f"주간 지수가 공표 후 {int(worst)}일 지난 값 — index-weekly 수집 확인 필요")
    return value


def compute(query: pd.DataFrame, reference: pd.DataFrame, weekly_index: pd.DataFrame, date_col: str = "계약일자") -> pd.DataFrame:
    """query: 단지일련번호 · 전용면적 · 시군구명 · date_col (질의일). 반환: query와 같은 인덱스 · 순서의 7개 열."""
    wk = prepare_index(weekly_index)
    q = pd.DataFrame({
        "_pos": np.arange(len(query)),
        "단지일련번호": query["단지일련번호"].astype(str).to_numpy(),
        "면적키": query["전용면적"].astype(float).round(0).to_numpy(),
        "질의일": pd.to_datetime(query[date_col]).astype("datetime64[ns]").to_numpy(),  # 1행 질의(초 단위 Timestamp)도 참고 표와 같은 단위로
    })
    last = pd.merge_asof(q.sort_values("질의일"), reference, left_on="질의일", right_on="공개일",
                         by=["단지일련번호", "면적키"], direction="backward").sort_values("_pos")
    out = pd.DataFrame(index=query.index)
    out["직전거래가_동일면적"] = last["거래금액"].astype(float).to_numpy()
    out["직전3건중위_동일면적"] = last["직전3건중위"].astype(float).to_numpy()
    out["직전거래_경과일"] = (last["질의일"] - last["직전계약일"]).dt.days.to_numpy()
    out["직전거래_층"] = last["층"].astype(float).to_numpy()
    out["동일면적_누적거래수"] = last["누적거래수"].astype(float).to_numpy()

    qday = pd.to_datetime(query[date_col]).astype("datetime64[ns]").reset_index(drop=True)
    gu = query["시군구명"].reset_index(drop=True)
    prev_day = qday - pd.to_timedelta(out["직전거래_경과일"].to_numpy(), unit="D")
    idx_now = _gu_index(wk, qday, gu, published_before=True)
    idx_prev = _gu_index(wk, prev_day, gu, published_before=False)
    idx_prev[(prev_day < wk["주"].min()).to_numpy()] = np.nan
    ratio = idx_now / idx_prev
    out["직전거래후_구지수변화"] = np.log(ratio)
    out["직전거래가_시세보정"] = out["직전거래가_동일면적"].to_numpy() * ratio
    return out[COLUMNS]
