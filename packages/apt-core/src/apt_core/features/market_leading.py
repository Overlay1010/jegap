"""시장 선행 Feature 16개 (01_market_leading 셀 3 · 8 ~ 27 · 29 · 32 이식, 레지스트리 source = 01_market).

| 원천 | 단위 | 공표 시차 |
|---|---|---|
| 부동산원(R-ONE) 월간 매매 · 전세 수급동향(아파트) | 서울 5개 권역 → 소속 구에 붙임 | 2개월 (익월 15일 공표) |
| 부동산원 월간 매매 · 전세 가격지수(주택종합) · 공식 전월 증감률 | 구 | 2개월 |
| 한국은행 주택가격전망CSI · 소비자심리지수(CCSI) | 전국 | 1개월 (당월 하순 공표) |

가격지수 수준값은 리베이스(최신월 = 100)로 미래 상승폭이 섞여 쓰지 않고, 증감률 · 3개월 변화만 쓴다.
행을 월 순서로 shift하므로 구마다 월이 빠짐없이 이어져야 한다 (빠지면 `_check_months`가 실패).
"""

from __future__ import annotations

import pandas as pd

PUBLICATION_LAG = {"매매가격지수": 2, "전세가격지수": 2, "매매가격지수_전월증감률": 2, "전세가격지수_전월증감률": 2,
                   "매매수급지수": 2, "전세수급지수": 2, "주택가격전망CSI": 1, "CCSI": 1}
CHANGE_COLS = ["매매수급지수", "전세수급지수", "주택가격전망CSI", "CCSI"]
COLUMNS = ["매매수급지수_lag2", "전세수급지수_lag2", "주택가격전망CSI_lag1", "CCSI_lag1",
           "매매수급지수_diff1", "전세수급지수_diff1", "주택가격전망CSI_diff1", "CCSI_diff1",
           "매매가격지수_전월증감률_lag2", "전세가격지수_전월증감률_lag2", "매매가격지수_momentum3m", "전세가격지수_momentum3m",
           "매매수급지수_diff3m", "전세수급지수_diff3m", "주택가격전망CSI_diff3m", "CCSI_diff3m"]
SEOUL_REGION_MAP = {
    "종로구": "도심권", "중구": "도심권", "용산구": "도심권",
    "성동구": "동북권", "광진구": "동북권", "동대문구": "동북권", "중랑구": "동북권", "성북구": "동북권", "강북구": "동북권", "도봉구": "동북권", "노원구": "동북권",
    "은평구": "서북권", "서대문구": "서북권", "마포구": "서북권",
    "양천구": "서남권", "강서구": "서남권", "구로구": "서남권", "금천구": "서남권", "영등포구": "서남권", "동작구": "서남권", "관악구": "서남권",
    "서초구": "동남권", "강남구": "동남권", "송파구": "동남권", "강동구": "동남권",
}
SEOUL_REGIONS = ["도심권", "동북권", "서북권", "서남권", "동남권"]


def read_rone_csv(path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc, header=None)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩 확인 필요: {path}")


def _month(s):
    return pd.to_datetime(s.str.replace("년 ", "-", regex=False).str.replace("월", "", regex=False), format="%Y-%m").dt.to_period("M")


def clean_supply(raw: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """R-ONE 수급동향 원본(넓은 표, 1행 = '2021년 1월' …) → (기준월, 권역) × 값. 서울만."""
    date_cols = list(range(4, raw.shape[1]))
    date_map = {c: str(raw.iloc[0, c]).strip() for c in date_cols}
    d = raw.iloc[2:].rename(columns={1: "시도", 2: "대권역", 3: "권역"})[["시도", "대권역", "권역"] + date_cols]
    d = d.melt(id_vars=["시도", "대권역", "권역"], value_vars=date_cols, var_name="_c", value_name=value_name)
    d["기준월"] = _month(d["_c"].map(date_map))
    d[value_name] = pd.to_numeric(d[value_name], errors="coerce")
    return d[d["시도"] == "서울"].drop(columns="_c").reset_index(drop=True)


def clean_price(raw: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """R-ONE 가격지수 원본(달마다 '원자료' · '전기대비증감률' 두 열) → (기준월, 자치구) × 지수 · 증감률. 서울 구만."""
    parts = []
    for c in range(5, raw.shape[1]):
        t = raw.iloc[3:, [1, 2, 3, 4, c]].copy()
        t.columns = ["시도", "대권역", "권역", "지역", "값"]
        t["기준월"], t["지표구분"] = str(raw.iloc[0, c]).strip(), str(raw.iloc[2, c]).strip()
        parts.append(t)
    d = pd.concat(parts, ignore_index=True)
    d["기준월"] = _month(d["기준월"])
    d["값"] = pd.to_numeric(d["값"], errors="coerce")
    d = d[(d["시도"] == "서울") & d["지역"].astype(str).str.endswith("구")]
    d = (d.pivot_table(index=["기준월", "지역"], columns="지표구분", values="값", aggfunc="first").reset_index()
         .rename(columns={"지역": "자치구", "원자료": f"{prefix}가격지수", "전기대비증감률": f"{prefix}가격지수_전월증감률"}))
    d.columns.name = None
    return d


def clean_bok(raw: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """한국은행 엑셀(8행부터 월별, 둘째 열 값) → (기준월, 값)."""
    d = raw.iloc[7:, [0, 1]].copy()
    d.columns = ["기준월", value_name]
    d["기준월"] = pd.to_datetime(d["기준월"], format="%Y-%m", errors="coerce").dt.to_period("M")
    d[value_name] = pd.to_numeric(d[value_name], errors="coerce")
    return d.dropna().reset_index(drop=True)


def base_table(sale_supply, jeonse_supply, sale_price, jeonse_price, housing_csi, ccsi, start="2021-01") -> pd.DataFrame:
    """정제한 6개 표 → (자치구, 기준월) 원자료. start는 시차 · 3개월 변화 계산에 필요한 첫 달 (V2: 2021-01)."""
    sup = (sale_supply[sale_supply["권역"].isin(SEOUL_REGIONS)][["기준월", "권역", "매매수급지수"]]
           .merge(jeonse_supply[jeonse_supply["권역"].isin(SEOUL_REGIONS)][["기준월", "권역", "전세수급지수"]],
                  on=["기준월", "권역"], how="inner", validate="one_to_one"))
    price = sale_price.merge(jeonse_price, on=["기준월", "자치구"], how="inner", validate="one_to_one")
    price["권역"] = price["자치구"].map(SEOUL_REGION_MAP)
    if price["권역"].isna().any():
        raise ValueError("권역에 없는 구가 있음")
    senti = housing_csi.merge(ccsi, on="기준월", how="inner", validate="one_to_one")
    b = price.merge(sup, on=["기준월", "권역"], how="left").merge(senti, on="기준월", how="left")
    b = b[b["기준월"] >= pd.Period(start, freq="M")]
    return b.sort_values(["자치구", "기준월"]).reset_index(drop=True)


def _check_months(b):
    for gu, g in b.groupby("자치구"):
        m = g["기준월"]
        if not (m.diff().dropna().map(lambda x: x.n) == 1).all():
            raise ValueError(f"{gu}: 월이 빠지거나 겹침 — shift 계산이 틀어짐")


def compute(base: pd.DataFrame, lags: dict | None = None) -> pd.DataFrame:
    """반환: (기준월, 자치구, 권역) × 시장 선행 Feature 16개 (base의 모든 달)."""
    lags = PUBLICATION_LAG if lags is None else lags
    _check_months(base)
    f = base.copy()
    g = f.groupby("자치구")
    for col in ["매매수급지수", "전세수급지수", "주택가격전망CSI", "CCSI", "매매가격지수_전월증감률", "전세가격지수_전월증감률"]:
        f[f"{col}_lag{PUBLICATION_LAG[col]}"] = g[col].shift(lags[col])
    for col in CHANGE_COLS:
        f[f"{col}_diff1"] = g[col].shift(lags[col]) - g[col].shift(lags[col] + 1)
        f[f"{col}_diff3m"] = g[col].shift(lags[col]) - g[col].shift(lags[col] + 3)
    for col in ["매매가격지수", "전세가격지수"]:
        f[f"{col}_momentum3m"] = (g[col].shift(lags[col]) / g[col].shift(lags[col] + 3) - 1) * 100
    return f[["기준월", "자치구", "권역"] + COLUMNS]
