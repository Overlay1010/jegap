"""금리 Feature (06_feature_merge 셀 12 이식, 레지스트리 source = 06_rate).

시점 규칙: 기준금리는 결정 즉시 공개 → m-1월 값(V1 그대로, 이름 '_전월').
주담대 · 전세대출 가중평균금리(한국은행, 익월 말 공표) → m-2월 값(이름 '_최근공표').
"""

from __future__ import annotations

import pandas as pd

RATE_LAG = {"기준금리": 1, "주담대금리": 2, "전세대출금리": 2}
START = "2020-12-01"  # 대출금리 원자료 시작


def monthly_rates(base_rate_wide: pd.DataFrame, loan_rate_wide: pd.DataFrame) -> pd.DataFrame:
    """한국은행_기준금리.csv(열 = 연월) + mortgage_rent_loan_rate_monthly.csv(계정항목 × 연월) → 월별 표."""
    base = base_rate_wide.melt(var_name="연월", value_name="기준금리")
    base = base[base["연월"].astype(str).str.match(r"\d{4}[./-]\d{2}")].copy()
    base["거래월일자"] = pd.to_datetime(base["연월"].astype(str).str.replace(".", "-", regex=False)
                                   .str.replace("/", "-", regex=False) + "-01", errors="coerce")
    base["기준금리"] = pd.to_numeric(base["기준금리"], errors="coerce")
    month_cols = [c for c in loan_rate_wide.columns if str(c)[:4].isdigit()]
    loan = loan_rate_wide[["계정항목"] + month_cols].melt(id_vars="계정항목", var_name="연월", value_name="금리")
    loan["거래월일자"] = pd.to_datetime(loan["연월"].str.extract(r"(\d{4}/\d{2})")[0] + "/01", format="%Y/%m/%d")
    loan["금리"] = pd.to_numeric(loan["금리"], errors="coerce")
    loan = loan.pivot(index="거래월일자", columns="계정항목", values="금리").reset_index()
    loan.columns = [str(c).strip() for c in loan.columns]
    loan = loan.rename(columns={"주택담보대출": "주담대금리", "전세자금대출": "전세대출금리"})
    out = (base[["거래월일자", "기준금리"]].dropna(subset=["거래월일자"])
           .merge(loan, on="거래월일자", how="inner").sort_values("거래월일자").reset_index(drop=True))
    return out[out["거래월일자"] >= START].reset_index(drop=True)


def compute(rate_monthly: pd.DataFrame, lags: dict[str, int] | None = None) -> pd.DataFrame:
    """반환: 거래월일자 × 금리 Feature 9개."""
    lags = RATE_LAG if lags is None else lags
    out = rate_monthly[["거래월일자"]].copy()
    for col, lag in lags.items():
        s = rate_monthly[col]
        level = f"{col}_전월" if lag == 1 else f"{col}_최근공표"
        suffix = "" if lag == 1 or col == "기준금리" else "_최근공표"
        out[level] = s.shift(lag)
        out[f"{col}_3개월변화{suffix}"] = s.shift(lag) - s.shift(lag + 3)
        out[f"{col}_6개월변화{suffix}"] = s.shift(lag) - s.shift(lag + 6)
    return out
