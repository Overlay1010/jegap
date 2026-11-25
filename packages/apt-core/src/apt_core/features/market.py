"""구 · 서울 매매 시장 Feature (06_feature_merge 셀 3 · 5 이식, 레지스트리 source = 06_market(확정월)).

시점 규칙: m월 행은 m-2월까지 확정된 거래만 쓴다 (실거래 신고 기한 30일 → m월 1일에는 m-2월까지 확정, TRADE_LAG = 2).
V1 정의 그대로(구 · 서울 월 중위가격 · 중위평당가 · 거래량, 직전 3 · 6개월 월중위값의 중위, 거래량 합, 1개월 변화율)이고
시차만 1 → 2로 바꿨다. 열 이름의 '전월 · 과거N개월'은 시차 2에서 '확정월 · 확정N개월'이 된다.

입력 거래 풀 = 해제 제외 매매 거래 (시군구명, 거래월일자(월 1일), 거래금액(만원), 평당가).
V2 학습 때 풀 = 워밍업 원본(2020-01 ~ 2021-06, 해제 제외) + V1 정제본(2021-07 ~, 해제 · 좌표 실패 제외) — `notebook_pool` 참고.
"""

from __future__ import annotations

import pandas as pd

TRADE_LAG = 2
PYEONG = 3.3058

V1_COLUMNS = ["구_전월_중위가격", "구_전월_중위평당가", "구_전월_거래량", "구_과거3개월_중위가격",
              "구_과거6개월_중위가격", "구_과거3개월_거래량", "구_과거6개월_거래량", "구_전월_가격변화율",
              "서울_전월_중위가격", "서울_과거3개월_중위가격", "서울_과거6개월_중위가격",
              "서울_전월_거래량", "서울_과거3개월_거래량", "서울_전월_가격변화율"]


def lag_label(name: str, lag: int) -> str:
    if lag == 1:
        return name
    return name.replace("전월", "확정월").replace("과거3개월", "확정3개월").replace("과거6개월", "확정6개월")


COLUMNS = [lag_label(c, TRADE_LAG) for c in V1_COLUMNS]
POOL_COLS = ["법정동시군구코드", "시군구명", "법정동", "단지일련번호", "거래월일자", "거래금액", "평당가"]


def notebook_pool(warmup_raw: pd.DataFrame, v1_final: pd.DataFrame) -> pd.DataFrame:
    """06 셀 3의 거래 풀: 워밍업 원본(해제 제외) + V1 정제본. 구 이름은 V1의 시군구코드 ↔ 시군구명 표로 붙인다."""
    v1 = v1_final.copy()
    v1["거래월일자"] = pd.to_datetime(v1["거래월일자"])
    for c in ("시군구명", "법정동", "단지일련번호"):
        v1[c] = v1[c].astype(str)
    gu_name = v1.drop_duplicates("법정동시군구코드").set_index("법정동시군구코드")["시군구명"]
    warm = warmup_raw[warmup_raw["해제여부"].ne("O")].copy()
    warm["법정동시군구코드"] = warm["법정동시군구코드"].astype(str).str.zfill(5)
    warm["시군구명"] = warm["법정동시군구코드"].map(gu_name)
    warm["법정동"] = warm["법정동"].astype(str).str.strip()
    warm["단지일련번호"] = warm["단지일련번호"].astype(str)
    warm["거래월일자"] = pd.to_datetime(dict(year=warm["계약년도"], month=warm["계약월"], day=1))
    warm["거래금액"] = warm["거래금액"].astype(float)
    warm["평당가"] = warm["거래금액"] / (warm["전용면적"].astype(float) / PYEONG)
    if warm["시군구명"].isna().any():
        raise ValueError("워밍업 거래에 구 이름을 붙이지 못한 행이 있습니다")
    pool_v1 = v1[POOL_COLS].copy()
    pool_v1["거래금액"] = pool_v1["거래금액"].astype(float)
    return pd.concat([warm[POOL_COLS], pool_v1], ignore_index=True)


def monthly_market(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gu = (trades.groupby(["시군구명", "거래월일자"])
          .agg(월중위거래금액=("거래금액", "median"), 월중위평당가=("평당가", "median"), 월거래량=("거래금액", "size"))
          .reset_index())
    months = pd.date_range(trades["거래월일자"].min(), trades["거래월일자"].max(), freq="MS")
    full = pd.MultiIndex.from_product([sorted(gu["시군구명"].unique()), months], names=["시군구명", "거래월일자"])
    gu = gu.set_index(["시군구명", "거래월일자"]).reindex(full).reset_index()
    gu["월거래량"] = gu["월거래량"].fillna(0)
    seoul = (trades.groupby("거래월일자")
             .agg(서울_월중위가격=("거래금액", "median"), 서울_월거래량=("거래금액", "size"))
             .reindex(months).rename_axis("거래월일자").reset_index())
    seoul["서울_월거래량"] = seoul["서울_월거래량"].fillna(0)
    return gu, seoul


def _roll(s: pd.Series, lag: int, window: int, how: str) -> pd.Series:
    r = s.shift(lag).rolling(window, min_periods=window)
    return r.median() if how == "median" else r.sum()


def compute(trades: pd.DataFrame, lag: int = TRADE_LAG) -> pd.DataFrame:
    """반환: (시군구명, 거래월일자) × 14열. 거래월일자 m 행은 m-lag월까지의 거래만 쓴다."""
    gu, seoul = monthly_market(trades)
    out = gu[["시군구명", "거래월일자"]].copy()
    g = gu.groupby("시군구명")
    L = lambda n: lag_label(n, lag)  # noqa: E731
    out[L("구_전월_중위가격")] = g["월중위거래금액"].shift(lag)
    out[L("구_전월_중위평당가")] = g["월중위평당가"].shift(lag)
    out[L("구_전월_거래량")] = g["월거래량"].shift(lag)
    out[L("구_과거3개월_중위가격")] = g["월중위거래금액"].transform(lambda s: _roll(s, lag, 3, "median"))
    out[L("구_과거6개월_중위가격")] = g["월중위거래금액"].transform(lambda s: _roll(s, lag, 6, "median"))
    out[L("구_과거3개월_거래량")] = g["월거래량"].transform(lambda s: _roll(s, lag, 3, "sum"))
    out[L("구_과거6개월_거래량")] = g["월거래량"].transform(lambda s: _roll(s, lag, 6, "sum"))
    out[L("구_전월_가격변화율")] = g["월중위거래금액"].transform(lambda s: s.shift(lag) / s.shift(lag + 1) - 1)

    s = seoul[["거래월일자"]].copy()
    p, n = seoul["서울_월중위가격"], seoul["서울_월거래량"]
    s[L("서울_전월_중위가격")] = p.shift(lag)
    s[L("서울_과거3개월_중위가격")] = _roll(p, lag, 3, "median")
    s[L("서울_과거6개월_중위가격")] = _roll(p, lag, 6, "median")
    s[L("서울_전월_거래량")] = n.shift(lag)
    s[L("서울_과거3개월_거래량")] = _roll(n, lag, 3, "sum")
    s[L("서울_전월_가격변화율")] = p.shift(lag) / p.shift(lag + 1) - 1
    return out.merge(s, on="거래월일자", how="left")
