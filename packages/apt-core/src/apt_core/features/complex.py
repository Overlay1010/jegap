"""단지 Feature 21개 (05b_complex_attributes 셀 1 · 3 · 6 · 7 · 9 · 10 · 17 이식, 레지스트리 source = 05b_complex).

| 묶음 | 원천 | 시점 규칙 |
|---|---|---|
| 물리 속성 9개 (동수 · 세대수 · 층 · 승강기 · 주차 · 용적률 · 건폐율) | 건축물대장 표제부(공동주택) + 총괄표제부 | 고정 속성 |
| 면적 구성 5개 | 공시가격 전체 호 (필지별 가장 최근 공시연도) | 고정 속성 (V2 그대로) |
| 공시 4개 | 공시가격 호 단위 → 필지 · 연도 · 면적대 중위 ㎡당 가격 | Y년 공시는 Y년 7월 1일부터 |
| 재건축 3개 | 서울시 정비사업 추진현황 → 재건축 구역 ↔ 단지 필지 | 단계 고시일 < 계약일인 단계 중 가장 높은 것 |

단지 ↔ 외부 자료 연결 키 = 필지(lot: 시군구 5 + 법정동 5 + 본번 4 + 부번 4).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

AREA_BINS, AREA_LABELS = [0, 60, 85, 135, np.inf], ["≤60", "60~85", "85~135", ">135"]
RELEASE_MONTH = 7
ATTR_COLUMNS = ["동수", "세대수", "최고층", "평균층", "지하층", "승강기_100세대", "세대당_주차", "용적률", "건폐율"]
MIX_COLUMNS = ["전용_중위", "전용_IQR", "소형60이하_비중", "대형85초과_비중", "공시_호수_최근"]
OFFICIAL_COLUMNS = ["공시_m2당_단지", "공시_m2당_면적대", "공시기반_추정가", "공시_전년대비"]
STAGE_COLUMNS = ["재건축_단계", "재건축_구역", "재건축_조합설립이후"]
COLUMNS = OFFICIAL_COLUMNS + STAGE_COLUMNS + ATTR_COLUMNS + MIX_COLUMNS
STAGES = [("구역지정_최초", 1), ("추진위원회", 2), ("조합설립", 3), ("사업시행_최초", 4),
          ("관리처분_최초", 5), ("이주시작", 6), ("착공", 7)]
JEONGBI_COLS = ["CODE", "순번", "자치구", "구역명", "위치1", "위치2", "공공민간", "일반재촉", "사업유형", "사업추진단계",
                "기존가구수", "구역지정_최초", "구역지정_변경", "추진위원회", "조합설립", "건축심의", "사업시행_최초", "사업시행_변경",
                "관리처분_최초", "관리처분_변경", "이주시작", "이주종료", "착공", "건립세대_계", "분양", "임대"]
_PARK = ["indrAutoUtcnt", "oudrAutoUtcnt", "indrMechUtcnt", "oudrMechUtcnt"]


def lot_of(trades: pd.DataFrame) -> pd.Series:
    return (trades["법정동시군구코드"].astype(str).str.zfill(5) + trades["법정동읍면동코드"].astype(str).str.zfill(5)
            + trades["법정동본번코드"].astype(str).str.zfill(4) + trades["법정동부번코드"].astype(str).str.zfill(4))


def complexes_of(trades: pd.DataFrame) -> pd.DataFrame:
    """거래 표(base.clean 출력) → 단지 표. 단지마다 거래 순서상 첫 행 (05b 셀 1)."""
    t = trades.assign(lot=lot_of(trades))
    first = t.drop_duplicates("단지일련번호")
    return first[["단지일련번호", "시군구명", "lot"]].assign(
        법정동=first["법정동"].astype(str).str.strip(),  # V1 셀 42와 같이 공백 제거
        본번=first["법정동본번코드"].astype(int), 부번=first["법정동부번코드"].astype(int)).reset_index(drop=True)


# ---------- 물리 속성 (건축물대장) ----------
def _lot_key(d):
    return d["sigunguCd"] + d["bjdongCd"] + d["bun"].str.zfill(4) + d["ji"].str.zfill(4)


def _num(d, cols):
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    return d


def attributes(complexes: pd.DataFrame, title: pd.DataFrame, recap: pd.DataFrame) -> pd.DataFrame:
    """단지 × 물리 속성 9개 (+ 사용승인연도). title = 공동주택 표제부, recap = 총괄표제부."""
    title = _num(title.copy(), ["hhldCnt", "grndFlrCnt", "ugrndFlrCnt", "rideUseElvtCnt", "emgenUseElvtCnt", "vlRat", "bcRat"] + _PARK)
    title["lot"] = _lot_key(title)
    title["사용승인연도"] = pd.to_numeric(title["useAprDay"].astype(str).str[:4], errors="coerce")
    title["주차"] = title[_PARK].sum(axis=1)
    by_lot = title[title["hhldCnt"] > 0].groupby("lot").agg(
        동수=("hhldCnt", "size"), 대장세대수=("hhldCnt", "sum"), 최고층=("grndFlrCnt", "max"), 평균층=("grndFlrCnt", "mean"),
        지하층=("ugrndFlrCnt", "max"), 승강기=("rideUseElvtCnt", "sum"),
        표제_용적률=("vlRat", lambda s: s[s > 0].max() if (s > 0).any() else np.nan),
        표제_건폐율=("bcRat", lambda s: s[s > 0].max() if (s > 0).any() else np.nan),
        표제_주차=("주차", "sum"), 사용승인연도=("사용승인연도", "min"))
    recap = _num(recap.copy(), ["hhldCnt", "totPkngCnt", "vlRat", "bcRat", "mainBldCnt"] + _PARK)
    recap["lot"] = _lot_key(recap)
    recap["주차"] = recap["totPkngCnt"].where(recap["totPkngCnt"] > 0, recap[_PARK].sum(axis=1))
    by_recap = recap.groupby("lot").agg(총괄_주차=("주차", "max"), 총괄_세대수=("hhldCnt", "max"),
                                         총괄_용적률=("vlRat", "max"), 총괄_건폐율=("bcRat", "max"))
    a = complexes[["단지일련번호", "lot"]].merge(by_lot, left_on="lot", right_index=True, how="left") \
        .merge(by_recap, left_on="lot", right_index=True, how="left")
    # 단지 대표값: 총괄표제부 우선, 없으면 동별 표제부
    a["세대수"] = a["총괄_세대수"].where(a["총괄_세대수"] > 0, a["대장세대수"])
    a["주차대수"] = a["총괄_주차"].where(a["총괄_주차"] > 0, a["표제_주차"])
    a["세대당_주차"] = (a["주차대수"] / a["세대수"]).where(a["주차대수"] > 0)
    a["용적률"] = a["총괄_용적률"].where(a["총괄_용적률"] > 0, a["표제_용적률"])
    a["건폐율"] = a["총괄_건폐율"].where(a["총괄_건폐율"] > 0, a["표제_건폐율"])
    a["승강기_100세대"] = (a["승강기"] / a["대장세대수"] * 100).where(a["승강기"] > 0)
    # 비현실적인 값은 입력 오류로 보고 결측
    a.loc[~a["세대당_주차"].between(0.05, 5), "세대당_주차"] = np.nan
    a.loc[~a["용적률"].between(30, 1500), "용적률"] = np.nan
    a.loc[~a["건폐율"].between(5, 100), "건폐율"] = np.nan
    a.loc[a["최고층"] <= 0, ["최고층", "평균층"]] = np.nan
    return a[["단지일련번호", "lot", "사용승인연도"] + ATTR_COLUMNS]


# ---------- 공시가격 ----------
def official_units(official_raw: pd.DataFrame) -> pd.DataFrame:
    """공시가격 호 단위 원본(기준연도 · 법정동코드 · 본번 · 부번 · 전용면적 · 공시가격) → lot · ㎡당(만원) · 면적대."""
    g = official_raw[["기준연도", "법정동코드", "본번", "부번", "전용면적", "공시가격"]].copy()
    g["lot"] = g["법정동코드"].astype(str).str.zfill(10) + g["본번"].astype(str).str.zfill(4) + g["부번"].astype(str).str.zfill(4)
    g["전용면적"] = pd.to_numeric(g["전용면적"], errors="coerce")
    g["m2당"] = pd.to_numeric(g["공시가격"], errors="coerce") / g["전용면적"] / 10_000
    g["면적대"] = pd.cut(g["전용면적"], AREA_BINS, labels=AREA_LABELS)
    return g


def official_tables(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(필지 × 연도 ㎡당 중위 · 호 수, 필지 × 연도 × 면적대 ㎡당 중위)."""
    g = units.dropna(subset=["m2당"]).astype({"기준연도": int})
    price_lot = g.groupby(["lot", "기준연도"])["m2당"].agg(공시_m2당_단지="median", 공시_호수="size").reset_index()
    price_band = g.groupby(["lot", "기준연도", "면적대"], observed=True)["m2당"].median().rename("공시_m2당_면적대").reset_index()
    return price_lot, price_band


def unit_mix(units: pd.DataFrame, complexes: pd.DataFrame) -> pd.DataFrame:
    """단지 면적 구성: 공시가격에 잡힌 전체 호(필지별 가장 최근 공시연도) 기준."""
    u = units[units["lot"].isin(complexes["lot"])][["기준연도", "lot", "전용면적"]].dropna().astype({"기준연도": int})
    u = u[u["기준연도"].eq(u.groupby("lot")["기준연도"].transform("max"))]
    mix = u.groupby("lot").agg(
        면적기준연도=("기준연도", "first"), 전용_중위=("전용면적", "median"),
        전용_IQR=("전용면적", lambda s: s.quantile(0.75) - s.quantile(0.25)),
        소형60이하_비중=("전용면적", lambda s: (s <= 60).mean()), 대형85초과_비중=("전용면적", lambda s: (s > 85).mean()),
        공시_호수_최근=("전용면적", "size")).reset_index()
    return complexes[["단지일련번호", "lot"]].merge(mix, on="lot", how="left").drop(columns="lot")


def available_year(dates: pd.Series) -> np.ndarray:
    return np.where(dates.dt.month >= RELEASE_MONTH, dates.dt.year, dates.dt.year - 1)


def official_features(query: pd.DataFrame, price_lot: pd.DataFrame, price_band: pd.DataFrame) -> pd.DataFrame:
    """query: lot · 계약일자 · 전용면적. 반환: query 순서의 공시 4개 (+ 공시연도)."""
    t = query[["lot", "계약일자", "전용면적"]].reset_index(drop=True).copy()
    t["공시연도"] = available_year(pd.to_datetime(t["계약일자"]))
    t["면적대"] = pd.cut(t["전용면적"].astype(float), AREA_BINS, labels=AREA_LABELS)
    t = t.merge(price_lot, left_on=["lot", "공시연도"], right_on=["lot", "기준연도"], how="left").drop(columns="기준연도")
    prev = price_lot.assign(기준연도=price_lot["기준연도"] + 1).rename(columns={"공시_m2당_단지": "공시_m2당_전년"})
    t = t.merge(prev[["lot", "기준연도", "공시_m2당_전년"]], left_on=["lot", "공시연도"],
                right_on=["lot", "기준연도"], how="left").drop(columns="기준연도")
    t = t.merge(price_band, left_on=["lot", "공시연도", "면적대"], right_on=["lot", "기준연도", "면적대"],
                how="left").drop(columns="기준연도")
    band = t["공시_m2당_면적대"].fillna(t["공시_m2당_단지"])
    return pd.DataFrame({"공시연도": t["공시연도"], "공시_m2당_단지": t["공시_m2당_단지"], "공시_m2당_면적대": band,
                         "공시기반_추정가": band * t["전용면적"].astype(float),
                         "공시_전년대비": (t["공시_m2당_단지"] / t["공시_m2당_전년"] - 1) * 100})


# ---------- 재건축 단계 ----------
def read_jeongbi(path) -> pd.DataFrame:
    d = pd.read_excel(path, header=None, skiprows=4)
    d.columns = JEONGBI_COLS
    return d[d["CODE"].notna()].copy()


def _parse_lot(text, gu):
    # 지번 표기: '수송동146-12', '이촌동193-3,5'(여러 필지 → 첫 필지), '중곡4동18-24'(행정동), '양천구신정동1200'(구 이름 포함)
    s = str(text).replace(" ", "").replace("번지", "").replace(gu, "")
    m = re.match(r"^(.+?[동가로])(\d+)(?:-(\d+))?", s)
    if not m:
        return None
    dong = re.sub(r"\d+동$", "동", m.group(1))   # 중곡4동 → 중곡동 (행정동 → 법정동 근사)
    dong = re.sub(r"본동$", "동", dong)           # 망우본동 → 망우동
    return dong, int(m.group(2)), int(m.group(3) or 0)


def rebuild_zones(jeongbi: pd.DataFrame, complexes: pd.DataFrame) -> pd.DataFrame:
    """정비사업 추진현황 → 재건축 구역 ↔ 단지 (단지일련번호 · CODE · 단계별 고시일)."""
    z = jeongbi.copy()
    for col, _ in STAGES:
        z[col] = pd.to_datetime(z[col], errors="coerce")
    rb = z[z["사업유형"].isin(["공동주택재건축", "아파트지구재건축"])].copy()
    parsed = rb.apply(lambda r: _parse_lot(r["위치1"], r["자치구"]), axis=1)
    rb["법정동"] = parsed.map(lambda x: x[0] if x else None)
    rb["본번"] = parsed.map(lambda x: x[1] if x else None)
    rb["부번"] = parsed.map(lambda x: x[2] if x else None)
    link = rb.merge(complexes[["단지일련번호", "시군구명", "법정동", "본번", "부번"]],
                    left_on=["자치구", "법정동", "본번", "부번"], right_on=["시군구명", "법정동", "본번", "부번"], how="left")
    matched = link.dropna(subset=["단지일련번호"])
    return matched[["단지일련번호", "CODE"] + [c for c, _ in STAGES]].drop_duplicates("단지일련번호").reset_index(drop=True)


def stage_features(query: pd.DataFrame, zone_of: pd.DataFrame) -> pd.DataFrame:
    """query: 단지일련번호 · 계약일자. 고시일 < 계약일인 단계 중 가장 높은 단계."""
    t = query[["단지일련번호", "계약일자"]].reset_index(drop=True).merge(zone_of, on="단지일련번호", how="left")
    day = pd.to_datetime(t["계약일자"])
    stage = np.zeros(len(t), dtype=int)
    for col, value in STAGES:
        stage = np.where(t[col].notna() & (t[col] < day), np.maximum(stage, value), stage)
    return pd.DataFrame({"재건축_단계": stage, "재건축_구역": (stage >= 1).astype(int), "재건축_조합설립이후": (stage >= 3).astype(int)})


def compute(query: pd.DataFrame, price_lot, price_band, zone_of, attrs, mix) -> pd.DataFrame:
    """query: 단지일련번호 · lot · 계약일자 · 전용면적. 반환: query와 같은 인덱스의 21개 열."""
    q = query.reset_index(drop=True)
    out = pd.concat([official_features(q, price_lot, price_band)[OFFICIAL_COLUMNS], stage_features(q, zone_of)], axis=1)
    static = (q[["단지일련번호"]].merge(attrs[["단지일련번호"] + ATTR_COLUMNS], on="단지일련번호", how="left")
              .merge(mix[["단지일련번호"] + MIX_COLUMNS], on="단지일련번호", how="left"))
    out = pd.concat([out, static[ATTR_COLUMNS + MIX_COLUMNS]], axis=1)
    out.index = query.index
    return out[COLUMNS]
