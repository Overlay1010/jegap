"""인구 Feature 15개 + 구별 재고 (05_demographics 셀 3 · 7 · 9 · 17 이식, 레지스트리 source = 05_demographics).

원천 = KOSIS 표 7개 (주민등록 인구 · 세대 · 연령별 인구, 인구이동, 매입자 거주지 · 연령대, 주택총조사) — `data/external/kosis/*.parquet`.
시점 규칙: 데이터월 M → M+2월 행부터 (공표 M+1). 주택총조사 t년분 → t+1년 8월 행부터.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PUBLISH_LAG = 2
CENSUS_RELEASE_MONTH = 8
COLUMNS = ["인구_3m증감률", "인구_12m증감률", "세대_12m증감률", "세대원수_12m변화", "비중_30_44세_12m변화", "비중_65세이상_12m변화",
           "순이동률_3m", "순이동률_6m", "순이동률_12m", "타구전입_비중_3m", "외지인매입_비중_3m", "서울외매입_비중_3m", "30대매입_비중_3m",
           "서울_순이동률_12m", "서울_외지인매입_비중_3m"]
LEVEL_COLUMNS = ["인구", "세대", "세대원수", "비중_30_44세", "비중_65세이상"]
MIGRATION_ITEMS = {"T10": "총전입", "T20": "총전출", "T25": "순이동", "T31": "타구전입", "T32": "타구전출"}


def _monthly(d, value_name, **filters):
    x = d.copy()
    for col, val in filters.items():
        x = x[x[col].eq(val)]
    x = x.assign(지역=x["C1_NM"].replace({"서울특별시": "서울"}), data_month=pd.PeriodIndex(x["PRD_DE"], freq="M"),
                 value=pd.to_numeric(x["DT"], errors="coerce"))
    return x.groupby(["지역", "data_month"])["value"].sum().rename(value_name)


def base_table(raw: dict) -> pd.DataFrame:
    """KOSIS 원본 표들(키 = 표 ID) → (지역 = 서울 + 25개 구, 데이터월) × 원수치."""
    age = raw["DT_1B04006"]
    age = age[age["C2_NM"].ne("계")].copy()
    age["나이"] = age["C2_NM"].str.extract(r"(\d+)")[0].astype(int)
    mig = raw["DT_1B26001_A01"]
    res = raw["DT_408_2006_S0065"]
    res = res[res["ITM_NM"].eq("호수")]
    ba = raw["DT_408_2006_S0077"]
    ba = ba[ba["ITM_NM"].eq("호수")]
    buyer = pd.concat([_monthly(res, "매입_계", C2_NM="매입자거주지"), _monthly(res, "매입_구내", C2_NM="관할 시군구내"),
                       _monthly(res, "매입_서울타구", C2_NM="관할 시도내"), _monthly(res, "매입_서울외", C2_NM="관할 시도외 기타")], axis=1)
    buyer = buyer.join(pd.concat([_monthly(ba, "매입_연령계", C2_NM="매입자연령대"), _monthly(ba, "매입_30대", C2_NM="30대"),
                                  _monthly(ba, "매입_40대", C2_NM="40대")], axis=1), how="left")
    migration = pd.concat([_monthly(mig, n, ITM_ID=c) for c, n in MIGRATION_ITEMS.items()], axis=1)
    return pd.concat([_monthly(raw["DT_1B040A3"], "인구", ITM_ID="T20"), _monthly(raw["DT_1B040B3"], "세대", ITM_ID="T1"),
                      _monthly(age, "연령합계"), _monthly(age[age["나이"].between(30, 44)], "인구_30_44"),
                      _monthly(age[age["나이"] >= 65], "인구_65이상"), migration, buyer], axis=1).sort_index()


def _features(base):
    f = pd.DataFrame(index=base.index)
    g = lambda s, fn: s.groupby(level="지역", group_keys=False).apply(fn)  # noqa: E731
    shift = lambda col, k: g(base[col], lambda x: x.shift(k))  # noqa: E731
    rsum = lambda col, w: g(base[col], lambda x: x.rolling(w, min_periods=w).sum())  # noqa: E731
    f["인구_3m증감률"] = (base["인구"] / shift("인구", 3) - 1) * 100
    f["인구_12m증감률"] = (base["인구"] / shift("인구", 12) - 1) * 100
    f["세대_12m증감률"] = (base["세대"] / shift("세대", 12) - 1) * 100
    size = base["인구"] / base["세대"]
    f["세대원수_12m변화"] = size - g(size, lambda x: x.shift(12))
    s3044, s65 = base["인구_30_44"] / base["인구"] * 100, base["인구_65이상"] / base["인구"] * 100
    f["비중_30_44세_12m변화"] = s3044 - g(s3044, lambda x: x.shift(12))
    f["비중_65세이상_12m변화"] = s65 - g(s65, lambda x: x.shift(12))
    for w in (3, 6, 12):
        f[f"순이동률_{w}m"] = rsum("순이동", w) / base["인구"] * 100
    f["타구전입_비중_3m"] = rsum("타구전입", 3) / rsum("총전입", 3)
    f["외지인매입_비중_3m"] = (rsum("매입_서울타구", 3) + rsum("매입_서울외", 3)) / rsum("매입_계", 3)
    f["서울외매입_비중_3m"] = rsum("매입_서울외", 3) / rsum("매입_계", 3)
    f["30대매입_비중_3m"] = rsum("매입_30대", 3) / rsum("매입_연령계", 3)
    f["인구"], f["세대"], f["세대원수"], f["비중_30_44세"], f["비중_65세이상"] = base["인구"], base["세대"], size, s3044, s65
    return f


def compute(base: pd.DataFrame, months: pd.PeriodIndex, lag: int = PUBLISH_LAG) -> pd.DataFrame:
    """반환: (자치구, month) × 인구 Feature 15개 + 수준값 5개. m월 행 = 데이터월 m-lag."""
    f = _features(base).reset_index()
    f["month"] = f["data_month"] + lag
    seoul = f[f["지역"].eq("서울")].set_index("month")
    gu = f[f["지역"].ne("서울")].rename(columns={"지역": "자치구"})
    gu = gu[gu["month"].isin(months)].copy()
    gu["서울_순이동률_12m"] = gu["month"].map(seoul["순이동률_12m"])
    gu["서울_외지인매입_비중_3m"] = gu["month"].map(seoul["서울외매입_비중_3m"])
    return gu[["자치구", "month", "data_month"] + COLUMNS + LEVEL_COLUMNS].reset_index(drop=True)


def housing_stock(census_raw: pd.DataFrame, all_gu, months: pd.PeriodIndex) -> pd.DataFrame:
    """주택총조사(DT_1JU1501) → (자치구, month)마다 그때 공표돼 있던 가장 최근 조사의 주택 · 아파트 재고."""
    c = census_raw[census_raw["C1_NM"].isin(all_gu) & census_raw["C1"].str.match(r"^11\d{3}$")].copy()
    c["value"] = pd.to_numeric(c["DT"], errors="coerce")
    st = c.pivot_table(index=["C1_NM", "PRD_DE"], columns="ITM_NM", values="value").reset_index()
    st.columns.name = None
    st = st.rename(columns={"C1_NM": "자치구", "PRD_DE": "조사연도", "주택_계": "주택재고", "아파트": "아파트재고"})
    st["조사연도"] = st["조사연도"].astype(int)
    st["available_from"] = pd.PeriodIndex((st["조사연도"] + 1).astype(str) + f"-{CENSUS_RELEASE_MONTH:02d}", freq="M")
    if not st.groupby("조사연도")["자치구"].nunique().eq(len(all_gu)).all():
        raise ValueError("주택총조사에 빠진 구가 있음")
    grid = pd.DataFrame([(g, m) for g in all_gu for m in months], columns=["자치구", "month"])
    s, out = st.sort_values("available_from"), []
    for gu, part in grid.groupby("자치구"):
        hist = s[s["자치구"].eq(gu)]
        idx = np.searchsorted(hist["available_from"].to_numpy(), part["month"].to_numpy(), side="right") - 1
        if (idx < 0).any():
            raise ValueError(f"{gu}: 기준월보다 먼저 공표된 주택총조사가 없음")
        out.append(pd.concat([part.reset_index(drop=True), hist.iloc[idx].reset_index(drop=True)[["조사연도", "주택재고", "아파트재고"]]], axis=1))
    return pd.concat(out, ignore_index=True)
