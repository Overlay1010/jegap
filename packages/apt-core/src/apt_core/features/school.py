"""학교 Feature 23개 (02b_school_features 셀 3 · 5 ~ 12 · 16 이식, 레지스트리 source = 02_school 중 학교 부분).

입력 = 02 노트북의 EDSS 학교환경 표(학생수 · 위치 보정 완료) + EDSS 학생변동 · 학교개황 · 건물현황.
시점 규칙: 학교알리미 5월 공시 → 6월 이후 거래는 당해연도, 1 ~ 5월은 전년도 (EDSS 범위 밖은 가장 가까운 연도로 제한).
같은 좌표 · 같은 학교기준연도면 Feature가 같아서 좌표 × 연도 단위로 계산한 뒤 거래에 붙인다.
해당 연도에 운영 중인 학교만 쓰므로 폐교 · 신설이 자동 반영된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6_371_000
SCHOOL_RELEASE_MONTH = 6
MAX_SCHOOL_LAG_YEARS = 1  # 학습 데이터 최대 (2026 상반기 거래가 2025 자료). 넘으면 학교 통계 갱신이 밀린 것
NEAREST_LEVELS = {"초등학교": "최근접초", "중학교": "최근접중"}
NEAREST_VALUE_COLS = ["학생수", "학급당학생수", "교원1인당학생수", "학생수증감률", "학생순전입률",
                      "학생1인당_교수학습공간", "학생1인당_체육장면적", "교사동_평균경과년수"]
RADIUS_COUNT_RULES = [("초등학교", 500, "반경500m_초등학교수"), ("중학교", 1000, "반경1km_중학교수"),
                      ("고등학교", 1000, "반경1km_고등학교수")]
COLUMNS = ([f"{p}_{c}" for p in NEAREST_LEVELS.values() for c in ["거리_m"] + NEAREST_VALUE_COLS]
           + [n for _, _, n in RADIUS_COUNT_RULES] + ["반경1km_초등학생수", "반경1km_초등순전입률"])
DISPLAY = [f"{p}_학교명" for p in NEAREST_LEVELS.values()]
_MOVE = ["조사년도3월학생수", "입학증가수", "도내전입증가수", "도외전입증가수", "도내전출감소수", "도외전출감소수"]
_AREA = {"유초중등학교개황_교수학습공간면적": "교수학습공간면적", "유초중등학교개황_학교용지체육장면적": "체육장면적"}


def school_table(school_env: pd.DataFrame, movement: pd.DataFrame, overview: pd.DataFrame, buildings: pd.DataFrame) -> pd.DataFrame:
    """학교 × 기준연도 지표 표. movement · overview · buildings = EDSS 학생변동 · 학교개황 · 건물현황 (조사년도 · 개방ID)."""
    s = school_env.sort_values(["개방ID", "기준연도"]).reset_index(drop=True)
    prev = s.groupby("개방ID")[["기준연도", "학생수"]].shift(1)
    s["학생수증감률"] = np.where(prev["기준연도"] == s["기준연도"] - 1, (s["학생수"] / prev["학생수"] - 1) * 100, np.nan)

    m = movement[["조사년도", "개방ID"] + _MOVE].copy()
    m[_MOVE] = m[_MOVE].apply(pd.to_numeric, errors="coerce")
    m = m.groupby(["조사년도", "개방ID"])[_MOVE].sum().reset_index().rename(columns={"조사년도": "기준연도"})
    m["학년도시작학생수"] = m["조사년도3월학생수"] + m["입학증가수"]
    m["순전입학생수"] = m["도내전입증가수"] + m["도외전입증가수"] - m["도내전출감소수"] - m["도외전출감소수"]
    m["학생순전입률"] = np.where(m["학년도시작학생수"] > 0, m["순전입학생수"] / m["학년도시작학생수"] * 100, np.nan)

    area = list(_AREA.values())
    f = overview[["조사년도", "개방ID"] + list(_AREA)].rename(columns={"조사년도": "기준연도", **_AREA})
    f[area] = f[area].apply(pd.to_numeric, errors="coerce").replace(0, np.nan)
    f = f.sort_values(["개방ID", "기준연도"])
    f[area] = f.groupby("개방ID")[area].ffill()  # 2024년은 전 학교 면적이 0 → 같은 학교 직전 연도 값 (과거 값만)

    b = buildings[["조사년도", "개방ID", "건물용도명", "건축년도", "건물연면적"]].rename(columns={"조사년도": "기준연도"})
    b = b.assign(건축년도=pd.to_numeric(b["건축년도"], errors="coerce"), 건물연면적=pd.to_numeric(b["건물연면적"], errors="coerce"))
    b = b[(b["건물용도명"] == "교사") & b["건축년도"].between(1880, b["기준연도"]) & (b["건물연면적"] > 0)].copy()
    b["가중경과년수"] = (b["기준연도"] - b["건축년도"]) * b["건물연면적"]
    age = b.groupby(["기준연도", "개방ID"])[["가중경과년수", "건물연면적"]].sum().reset_index()
    age["교사동_평균경과년수"] = age["가중경과년수"] / age["건물연면적"]

    s = (s.merge(m[["기준연도", "개방ID", "학년도시작학생수", "순전입학생수", "학생순전입률"]], on=["기준연도", "개방ID"], how="left", validate="one_to_one")
         .merge(f[["기준연도", "개방ID"] + area], on=["기준연도", "개방ID"], how="left", validate="one_to_one")
         .merge(age[["기준연도", "개방ID", "교사동_평균경과년수"]], on=["기준연도", "개방ID"], how="left", validate="one_to_one"))
    s["학생1인당_교수학습공간"] = s["교수학습공간면적"] / s["학생수"]
    s["학생1인당_체육장면적"] = s["체육장면적"] / s["학생수"]
    return s


def school_year(dates: pd.Series, year_min: int, year_max: int) -> np.ndarray:
    d = pd.to_datetime(dates)
    return np.where(d.dt.month >= SCHOOL_RELEASE_MONTH, d.dt.year, d.dt.year - 1).clip(year_min, year_max)


def _tree(df):
    return BallTree(np.radians(df[["위도", "경도"]].to_numpy()), metric="haversine")


def location_features(locations: pd.DataFrame, schools: pd.DataFrame, year: int) -> pd.DataFrame:
    """좌표(위도 · 경도) × 한 학교기준연도 → 학교 Feature 23개 + 학교명."""
    sy = schools[schools["기준연도"] == year]
    rad = np.radians(locations[["위도", "경도"]].to_numpy())
    out = locations[["위도", "경도"]].copy()
    out["학교기준연도"] = year
    for level, prefix in NEAREST_LEVELS.items():
        s = sy[sy["학교급구분"] == level].reset_index(drop=True)
        dist, idx = _tree(s).query(rad, k=1)
        near = s.iloc[idx[:, 0]]
        out[f"{prefix}_학교명"] = near["학교명"].to_numpy()
        out[f"{prefix}_거리_m"] = dist[:, 0] * EARTH_RADIUS_M
        for c in NEAREST_VALUE_COLS:
            out[f"{prefix}_{c}"] = near[c].to_numpy()
    for level, radius, name in RADIUS_COUNT_RULES:
        out[name] = _tree(sy[sy["학교급구분"] == level]).query_radius(rad, r=radius / EARTH_RADIUS_M, count_only=True)
    elem = sy[sy["학교급구분"] == "초등학교"].reset_index(drop=True)
    nbrs = _tree(elem).query_radius(rad, r=1000 / EARTH_RADIUS_M)
    students = elem["학생수"].to_numpy()
    out["반경1km_초등학생수"] = [students[i].sum() for i in nbrs]
    has = elem["학년도시작학생수"].notna().to_numpy()  # 순전입 지표가 있는 학교만 분자 · 분모에
    net, base = elem["순전입학생수"].fillna(0).to_numpy(), elem["학년도시작학생수"].fillna(0).to_numpy()
    rates = []
    for i in nbrs:
        i = i[has[i]]
        rates.append(net[i].sum() / base[i].sum() * 100 if base[i].sum() > 0 else np.nan)
    out["반경1km_초등순전입률"] = rates
    return out


def compute(query: pd.DataFrame, schools: pd.DataFrame, date_col: str = "계약일자") -> pd.DataFrame:
    """query: 위도 · 경도 · date_col. 반환: query와 같은 인덱스의 학교 Feature 23개 + 학교기준연도 · 학교명."""
    d = pd.to_datetime(query[date_col])
    wanted = np.where(d.dt.month >= SCHOOL_RELEASE_MONTH, d.dt.year, d.dt.year - 1)
    if (lag := int(wanted.max() - schools["기준연도"].max())) > MAX_SCHOOL_LAG_YEARS:
        raise ValueError(f"학교 통계가 {lag}년 밀림 (최신 {int(schools['기준연도'].max())}년) — EDSS · 학교알리미 연간 갱신 필요")
    yr = school_year(query[date_col], int(schools["기준연도"].min()), int(schools["기준연도"].max()))
    q = pd.DataFrame({"위도": query["위도"].to_numpy(), "경도": query["경도"].to_numpy(), "학교기준연도": yr})
    locs = q[["위도", "경도"]].drop_duplicates().reset_index(drop=True)
    feats = pd.concat([location_features(locs, schools, y) for y in sorted(q["학교기준연도"].unique())], ignore_index=True)
    out = q.merge(feats, on=["위도", "경도", "학교기준연도"], how="left", validate="many_to_one")
    out.index = query.index
    return out[COLUMNS + ["학교기준연도"] + DISPLAY]


# ---------------- EDSS 원본 → 학교 × 기준연도 표 (02_school_environment 셀 33 ~ 35 · 38 ~ 42 · 55 · 59 ~ 62 이식, 설계 1-4b R9) ----------------
# 관리하는 표 (학교알리미 API로 만든 것, 새 학교가 생기면 사람이 추가):
#   학교코드 대응표 school_crosswalk_2021_2025 (개방ID → SCHUL_CODE · 학교명, 수동 보정 포함)
#   학교 좌표 = 학교 코드맵 school_code_map_v2 + 보충표 data/external/school_location_patch.csv (코드맵에 없는 9곳)
#   학교알리미 학생수 school_environment_2024_2026 (방송통신 부설학교 학생수 보정용)
EDSS_FILE_GROUPS = {"학교속성": "0001. 공통_학교속성", "학교개황": "0002. 유초중등학교개황", "학급현황": "0003. 유초중등학급현황",
                    "교원현황": "0007. 유초중등교원현황", "도서관운영현황": "0008. 도서관운영현황", "건물현황": "0105. 건물현황",
                    "학생변동현황": "0210. 학생변동현황"}
HIGH_SCHOOL_TYPES = {"일반고등학교", "특성화고등학교", "특수목적고등학교", "자율고등학교"}
BROADCAST_GAP = 50  # EDSS 학생수가 학교알리미보다 50명 넘게 많으면 부설 방송통신중 · 고 학생이 합산된 학교
SCHOOL_LOCATION_OVERRIDES = [  # 덕수고: 2024년 위례 이전 때 개방ID가 캠퍼스 간 바뀜 → 2021 ~ 2023 행은 성동 옛 캠퍼스 좌표
    {"개방ID": 9478733062, "시작연도": 2021, "종료연도": 2023, "위도": 37.5552642216916, "경도": 127.041141809583}]
TABLE_COLUMNS = ["기준연도", "개방ID", "SCHUL_CODE", "학교명", "자치구", "학교급구분", "설립형태", "학생수", "학생수_원본", "학생수_보정여부",
                 "학급수", "교원수", "학급당학생수", "교원1인당학생수", "위도", "경도", "위치보정여부"]


def _read_edss_csv(path):
    for enc in ["cp949", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩 확인 필요: {path}")


def merge_edss(paths) -> dict[str, pd.DataFrame]:
    """EDSS 연도별 CSV('0001. 공통_학교속성(21).csv' 등) → 종류별 통합 표 7개. 종류마다 2021 ~ 2025 다섯 해가 있어야 한다."""
    import re
    from pathlib import Path
    out = {}
    for group, prefix in EDSS_FILE_GROUPS.items():
        pattern = re.compile(rf"{re.escape(prefix)}\((\d{{2}})\)\.csv$")
        files = sorted(p for p in map(Path, paths) if pattern.fullmatch(p.name))
        if len(files) != 5:
            raise ValueError(f"{prefix}: 파일 {len(files)}개 (5개 필요)")
        d = pd.concat([_read_edss_csv(p) for p in files], ignore_index=True, sort=False)
        keys = [c for c in ["조사년도", "개방ID"] if c in d.columns]
        out[group] = d.sort_values(keys).reset_index(drop=True) if keys else d
    return out


def _level(t):
    return "초등학교" if t == "초등학교" else "중학교" if t == "중학교" else "고등학교" if t in HIGH_SCHOOL_TYPES else pd.NA


def edss_school_table(attribute, overview, class_status, crosswalk, schoolinfo, code_map, location_patch) -> pd.DataFrame:
    base = attribute[attribute["시도명"] == "서울"].copy()
    base["학교급구분"] = base["학제유형명"].apply(_level)
    base = (base[base["학교급구분"].notna()][["조사년도", "개방ID", "시군명", "학교급구분", "설립구분명"]]
            .rename(columns={"조사년도": "기준연도", "시군명": "자치구", "설립구분명": "설립형태"}).reset_index(drop=True))
    ov = overview[["조사년도", "개방ID", "유초중등학교개황_학생수", "유초중등학교개황_교원수"]].rename(
        columns={"조사년도": "기준연도", "유초중등학교개황_학생수": "학생수", "유초중등학교개황_교원수": "교원수"})
    cl = class_status[["조사년도", "개방ID", "유초중등학급_학급수"]].rename(columns={"조사년도": "기준연도", "유초중등학급_학급수": "학급수"})
    e = (base.merge(ov, on=["기준연도", "개방ID"], how="left", validate="one_to_one")
         .merge(cl, on=["기준연도", "개방ID"], how="left", validate="one_to_one"))
    e[["학생수", "학급수", "교원수"]] = e[["학생수", "학급수", "교원수"]].apply(pd.to_numeric, errors="coerce")
    e = e[(e["학생수"] > 0) & (e["학급수"] > 0) & (e["교원수"] > 0)].reset_index(drop=True)  # 휴교 등 0인 학교 제외
    e = e.merge(crosswalk, on="개방ID", how="left", validate="many_to_one")
    if e["SCHUL_CODE"].isna().any():
        raise ValueError(f"학교코드 대응표에 없는 EDSS 학교: {sorted(e.loc[e['SCHUL_CODE'].isna(), '개방ID'].unique())[:5]} — 대응표 추가 필요")

    # 방송통신 부설학교: 2024 · 2025는 학교알리미 학생수로, 2021 ~ 2023은 두 해 평균 차이를 빼서 추정
    si = schoolinfo[schoolinfo["기준연도"].isin([2024, 2025])][["기준연도", "SCHUL_CODE", "학생수", "학급수"]]
    gap = e[e["기준연도"].isin([2024, 2025])][["기준연도", "개방ID", "SCHUL_CODE", "학생수", "학급수"]].merge(
        si, on=["기준연도", "SCHUL_CODE"], how="inner", suffixes=("_EDSS", "_SchoolInfo"), validate="one_to_one")
    gap["학생수차이"] = gap["학생수_EDSS"] - gap["학생수_SchoolInfo"]
    gap = gap[gap["학생수차이"] > BROADCAST_GAP]
    offset = gap.groupby("개방ID")["학생수차이"].mean()
    e["학생수_원본"] = e["학생수"]
    e["학생수_보정여부"] = e["개방ID"].isin(offset.index)
    est = e["학생수_원본"] - e["개방ID"].map(offset).fillna(0)
    si_students = e[["기준연도", "개방ID"]].merge(gap[["기준연도", "개방ID", "학생수_SchoolInfo"]], on=["기준연도", "개방ID"], how="left")["학생수_SchoolInfo"]
    e["학생수"] = si_students.fillna(pd.Series(est.to_numpy(), index=si_students.index)).round().astype(int).to_numpy()
    e["학급당학생수"] = e["학생수"] / e["학급수"]
    e["교원1인당학생수"] = e["학생수"] / e["교원수"]

    loc = pd.concat([code_map[["SCHUL_CODE", "위도", "경도"]], location_patch[["SCHUL_CODE", "위도", "경도"]]], ignore_index=True).drop_duplicates("SCHUL_CODE")
    e = e.merge(loc, on="SCHUL_CODE", how="left")
    e["위치보정여부"] = False
    for o in SCHOOL_LOCATION_OVERRIDES:
        mask = (e["개방ID"] == o["개방ID"]) & e["기준연도"].between(o["시작연도"], o["종료연도"])
        e.loc[mask, ["위도", "경도"]] = [o["위도"], o["경도"]]
        e.loc[mask, "위치보정여부"] = True
    if e[["위도", "경도"]].isna().any().any():
        raise ValueError(f"좌표 없는 학교: {sorted(e.loc[e['위도'].isna(), 'SCHUL_CODE'].unique())[:5]} — 학교 좌표 보충표 추가 필요")
    out = e[TABLE_COLUMNS].sort_values(["기준연도", "개방ID"]).reset_index(drop=True)
    if out.duplicated(["기준연도", "개방ID"]).any() or out.duplicated(["기준연도", "SCHUL_CODE"]).any():
        raise ValueError("학교 표 키 중복")
    return out
