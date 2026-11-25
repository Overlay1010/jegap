"""공급 모델 Feature 20개 = 서울 · 구 공급 14개 + 재고 대비 6개 (구 공급 원수치 6개는 비율 계산용, 모델 입력 아님) (04_housing_supply 셀 7 · 14 · 17 · 26, 06 셀 18 이식, 레지스트리 source = 04_supply · 06_supply_ratio).

| 묶음 | 원천 | 시점 규칙 |
|---|---|---|
| 서울 아파트 인허가 6 · 12m, 준공 3 · 6 · 12m | 국토부 주택건설실적 월별 최초 공표값 (`molit_seoul_permit_completion.csv`) | 사건월 M → M+2월 행부터 |
| 미분양 · 준공후미분양 (구 · 서울), 서울 3개월 변화, 미분양 있음 | 국토부 미분양 월별 파일 (파싱 결과 `unsold_seoul_gu_monthly.parquet`) | 같음 |
| 분양공고 3 · 6 · 12m, 입주예정 24m (구 · 서울), 공공분양 12m (서울) | 청약홈 APT 분양공고 | 공고일 < m월 1일 |
| 구 공급 ÷ 아파트 재고 × 1000 | 위 구 값 ÷ `housing_stock_v2` 아파트재고 | 재고 t년 → t+1년 8월 (05에서 결정) |

주택인허가 원장 기반 승인 · 준공 · 파이프라인은 V2에서 모델 제외(원장 준공이 공식 통계와 불일치, 원장 입력 지연)라 옮기지 않았다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PUBLISH_LAG = 2
OFFICIAL_COLUMNS = ["인허가아파트_서울_6m", "인허가아파트_서울_12m", "준공아파트_서울_3m", "준공아파트_서울_6m", "준공아파트_서울_12m"]
UNSOLD_COLUMNS = ["미분양_구", "준공후미분양_구", "미분양_있음", "미분양_서울", "준공후미분양_서울", "미분양_서울_3m변화"]
ANNOUNCE_COLUMNS = ["분양공고세대_3m", "분양공고세대_6m", "분양공고세대_12m", "입주예정_공고기준_24m",
                    "분양공고세대_3m_서울", "분양공고세대_6m_서울", "분양공고세대_12m_서울", "입주예정_공고기준_24m_서울", "공공분양세대_12m_서울"]
COLUMNS = OFFICIAL_COLUMNS + UNSOLD_COLUMNS + ANNOUNCE_COLUMNS
GU_COUNTS = ["분양공고세대_3m", "분양공고세대_6m", "분양공고세대_12m", "입주예정_공고기준_24m", "미분양_구", "준공후미분양_구"]
RATIO_COLUMNS = [f"{c}_재고천호당" for c in GU_COUNTS]


def official_monthly(series: pd.DataFrame) -> pd.DataFrame:
    """주택건설실적 표 → (kind, ref_month, apartment) 월별 최초 공표값."""
    o = series[series["basis"].eq("monthly_file")].copy()
    o["ref_month"] = pd.PeriodIndex(o["year"].astype(int).astype(str) + "-" + o["month"].astype(int).astype(str), freq="M")
    o = o[["kind", "ref_month", "apartment", "total"]].sort_values(["kind", "ref_month"])
    if o.duplicated(["kind", "ref_month"]).any():
        raise ValueError("같은 달 공식 실적이 두 번 있음")
    return o


def official_features(official: pd.DataFrame, months: pd.PeriodIndex, lag: int = PUBLISH_LAG) -> pd.DataFrame:
    out = pd.DataFrame({"month": months})
    for kind, name, windows in [("인허가", "인허가아파트_서울", (6, 12)), ("준공", "준공아파트_서울", (3, 6, 12))]:
        s = official[official["kind"].eq(kind)].set_index("ref_month")["apartment"].sort_index()
        s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
        for w in windows:
            r = s.rolling(w, min_periods=w).sum()
            r.index = r.index + lag
            out[f"{name}_{w}m"] = out["month"].map(r)
    return out


def unsold_features(unsold: pd.DataFrame, lag: int = PUBLISH_LAG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(구 × 월 미분양, 월 서울 미분양 + 3개월 변화). m월 행 = (m-lag)월 기준 값."""
    u = unsold.copy()
    u["ref_month"] = pd.PeriodIndex(u["ref_month"].astype(str), freq="M")
    u["month"] = u["ref_month"] + lag
    gu = u[u["구"].ne("계")].rename(columns={"구": "자치구", "미분양": "미분양_구", "준공후미분양": "준공후미분양_구"})
    se = u[u["구"].eq("계")].rename(columns={"미분양": "미분양_서울", "준공후미분양": "준공후미분양_서울"}).sort_values("month")
    se["미분양_서울_3m변화"] = se["미분양_서울"] - se["미분양_서울"].shift(3)
    return gu[["자치구", "month", "미분양_구", "준공후미분양_구"]], se[["month", "미분양_서울", "준공후미분양_서울", "미분양_서울_3m변화"]]


def prepare_announcements(raw: pd.DataFrame, all_gu) -> pd.DataFrame:
    ann = raw.copy()
    ann["공고일"] = pd.to_datetime(ann["RCRIT_PBLANC_DE"], errors="coerce")
    ann["세대수"] = pd.to_numeric(ann["TOT_SUPLY_HSHLDCO"], errors="coerce").fillna(0)
    ann["자치구"] = ann["HSSPLY_ADRES"].str.extract(r"서울(?:특별)?시\s*(\S+구)")[0]
    ann["공공"] = ann["HOUSE_DTL_SECD"].eq("03") | ann["HOUSE_SECD"].eq("10")  # 국민 · 신혼희망타운
    ann["입주예정월"] = pd.PeriodIndex(ann["MVN_PREARNGE_YM"].str[:4] + "-" + ann["MVN_PREARNGE_YM"].str[4:6], freq="M")
    if not (ann["자치구"].isin(all_gu).all() and ann["공고일"].notna().all()):
        raise ValueError("분양공고에 구 · 공고일을 읽지 못한 행이 있음")
    return ann


def announcement_features(ann: pd.DataFrame, months: pd.PeriodIndex, all_gu) -> pd.DataFrame:
    out = []
    for m in months:
        known = ann[ann["공고일"] < m.to_timestamp()]
        rec = {}
        for w in (3, 6, 12):
            win = known[known["공고일"] >= (m - w).to_timestamp()]
            rec.update({(g, f"분양공고세대_{w}m"): v for g, v in win.groupby("자치구")["세대수"].sum().items()})
            rec[("서울", f"분양공고세대_{w}m")] = win["세대수"].sum()
        win12 = known[known["공고일"] >= (m - 12).to_timestamp()]
        rec[("서울", "공공분양세대_12m")] = win12.loc[win12["공공"], "세대수"].sum()
        future = known[(known["입주예정월"] > m) & (known["입주예정월"] <= m + 24)]
        rec.update({(g, "입주예정_공고기준_24m"): v for g, v in future.groupby("자치구")["세대수"].sum().items()})
        rec[("서울", "입주예정_공고기준_24m")] = future["세대수"].sum()
        for gu in all_gu:
            r = {"자치구": gu, "month": m}
            for name in ["분양공고세대_3m", "분양공고세대_6m", "분양공고세대_12m", "입주예정_공고기준_24m"]:
                r[name] = rec.get((gu, name), 0)
                r[name + "_서울"] = rec.get(("서울", name), 0)
            r["공공분양세대_12m_서울"] = rec[("서울", "공공분양세대_12m")]
            out.append(r)
    return pd.DataFrame(out)


def compute(grid: pd.DataFrame, official: pd.DataFrame, unsold: pd.DataFrame, announcements: pd.DataFrame,
            housing_stock: pd.DataFrame, lag: int = PUBLISH_LAG) -> pd.DataFrame:
    """grid: 기준월('YYYY-MM') · 자치구. 반환: grid 순서의 공급 20개 + 재고 대비 6개."""
    all_gu = sorted(grid["자치구"].unique())
    months = pd.PeriodIndex(grid["기준월"].unique(), freq="M")
    g = grid[["기준월", "자치구"]].copy()
    g["month"] = pd.PeriodIndex(g["기준월"], freq="M")
    gu_u, se_u = unsold_features(unsold, lag)
    f = (g.merge(gu_u, on=["자치구", "month"], how="left").merge(se_u, on="month", how="left")
         .merge(official_features(official, months, lag), on="month", how="left")
         .merge(announcement_features(announcements, months, all_gu), on=["자치구", "month"], how="left"))
    f["미분양_있음"] = (f["미분양_구"] > 0).astype(int)
    f = f.merge(housing_stock[["기준월", "자치구", "아파트재고"]], on=["기준월", "자치구"], how="left", validate="one_to_one")
    for c in GU_COUNTS:
        f[f"{c}_재고천호당"] = f[c] / f["아파트재고"] * 1000
    f.index = grid.index
    return f[["기준월", "자치구"] + COLUMNS + RATIO_COLUMNS]


# ---------------- 미분양 원본 엑셀 파싱 (04 셀 14 이식, 설계 1-4b R8) ----------------
# - '시군구별' 시트는 마지막 열이 그 달 값 (머리글은 오타가 있어 쓰지 않음: 2019-10 파일의 '19.9 등)
# - 2019 ~ 2021년 파일은 0을 '-' 또는 빈칸으로 표기 → 0 (서울 계 = 25개 구 합으로 검증)
# - 그 달 파일의 마지막 열이 통째로 비어 있으면(2026-02) 다음 달 파일의 끝에서 두 번째 열을 쓰고 표시
def _seoul_rows(sheet):
    d = sheet.copy()
    d[0] = d[0].ffill()
    d = d[d[0].astype(str).str.replace(" ", "").str.contains("서울", na=False)].copy()
    d["구"] = d[1].astype(str).str.replace(" ", "")
    return d


def _to_number(s):
    return pd.to_numeric(s.astype(str).str.strip().replace({"-": "0", "nan": "0", "": "0"}), errors="coerce")


def _read_month(path, offset=0):
    book = pd.ExcelFile(path)
    sheet = pd.read_excel(book, "시군구별★", header=None)
    rows = _seoul_rows(sheet)
    raw = rows[sheet.columns[-1 - offset]]
    return rows["구"].to_numpy(), raw.notna().any(), _to_number(raw).to_numpy(), book


def parse_unsold_month(ref: pd.Period, files: dict) -> pd.DataFrame:
    gu, has_value, values, book = _read_month(files[ref])
    source = "당월 파일"
    if not has_value and (ref + 1) in files:
        gu, has_value, values, _ = _read_month(files[ref + 1], offset=1)
        source = "다음 달 파일"
    done = _seoul_rows(pd.read_excel(book, "공사완료후★", header=None))
    return pd.DataFrame({"ref_month": str(ref), "구": gu, "미분양": values, "미분양_출처": source}).merge(
        pd.DataFrame({"구": done["구"].to_numpy(), "준공후미분양": _to_number(done[2]).to_numpy()}), on="구", how="outer")


def parse_unsold(paths) -> pd.DataFrame:
    """국토부 미분양 월별 엑셀(unsold_YYYYMM.xlsx) 목록 → 구 × 월 (+ '계' = 서울 합)."""
    import re
    from pathlib import Path
    files = {pd.Period(re.search(r"unsold_(\d{6})", Path(p).name).group(1), freq="M"): p for p in paths}
    out = pd.concat([parse_unsold_month(ref, files) for ref in sorted(files)], ignore_index=True)
    se = out[out["구"].eq("계")].set_index("ref_month")["미분양"]
    gu_sum = out[out["구"].ne("계")].groupby("ref_month")["미분양"].sum()
    if not se.reindex(gu_sum.index).eq(gu_sum).all():
        raise ValueError("미분양 서울 계 ≠ 25개 구 합 — 엑셀 형식 변경 확인")
    return out
