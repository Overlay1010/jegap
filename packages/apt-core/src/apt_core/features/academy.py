"""학원 Feature 6개 (02c_academy_features 셀 3 ~ 7 이식, 레지스트리 source = 02_school 중 학원 부분).

입력 = 소상공인 상가정보 분기 파일의 교육 업종(P1) 추출본 (기준월 · 상권업종소분류코드 · 위도 · 경도 · 전체업소수 · 시군구명 · 행정동명).
시점 규칙: 스냅샷은 기준월 + 3개월부터 사용 가능 (공개 시차). 수집이 빠진 분기(입시교과 수가 전체 중앙값의 70% 미만)와
좌표 오류 분기는 건너뛰고, 그 거래월에 쓸 수 있는 정상 스냅샷 중 최신을 쓴다.
학원가 = 2020-12 스냅샷 입시 · 교과학원 DBSCAN(250m, 최소 80개, 학습 기간 이전), 학원 250개 이상 = 3대 학원가.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6_371_000
ACADEMY_GROUPS = {"입시교과": ["P10501"], "예체능": ["P10601", "P10609", "P10611", "P10613"], "외국어": ["P10615"]}
GAP_RATIO, RELEASE_LAG_MONTHS = 0.7, 3
MAX_SNAPSHOT_AGE = 26  # 학습 데이터 최대 경과개월 (결손 분기 때문). 넘으면 상가정보 분기 갱신이 밀린 것
SEOUL_LAT, SEOUL_LON = (37.41, 37.72), (126.76, 127.19)
CLUSTER_SNAPSHOT, CLUSTER_EPS_M, CLUSTER_MIN_SAMPLES, MAJOR_CLUSTER_SIZE = "202012", 250, 80, 250
RADIUS_RULES = [("입시교과", 500, "반경500m_입시교과학원수"), ("입시교과", 1000, "반경1km_입시교과학원수"),
                ("예체능", 1000, "반경1km_예체능학원수"), ("외국어", 1000, "반경1km_외국어학원수")]
COLUMNS = [n for _, _, n in RADIUS_RULES] + ["최근접학원가_거리_m", "최근접3대학원가_거리_m"]
DISPLAY = ["최근접학원가명", "최근접3대학원가명"]


def normal_snapshots(store_edu: pd.DataFrame) -> list[str]:
    s = store_edu.assign(입시교과=store_edu["상권업종소분류코드"].isin(ACADEMY_GROUPS["입시교과"])).groupby("기준월").agg(
        입시교과=("입시교과", "sum"), 위도_중앙값=("위도", "median"), 경도_중앙값=("경도", "median"))
    ok = (s["입시교과"] >= s["입시교과"].median() * GAP_RATIO) & s["위도_중앙값"].between(*SEOUL_LAT) & s["경도_중앙값"].between(*SEOUL_LON)
    return s.index[ok].tolist()


def _month_idx(yyyymm) -> np.ndarray:
    v = pd.Series(yyyymm).astype(int).to_numpy()
    return (v // 100) * 12 + (v % 100) - 1


def snapshot_for(month_yyyymm: pd.Series, snapshots: list[str]) -> pd.DataFrame:
    """거래연월(YYYYMM) → 사용 스냅샷 기준월 · 경과개월."""
    snap = pd.DataFrame({"기준월": snapshots, "기준월_idx": _month_idx(snapshots)})
    snap["사용가능_idx"] = snap["기준월_idx"] + RELEASE_LAG_MONTHS
    q = pd.DataFrame({"_pos": np.arange(len(month_yyyymm)), "거래월_idx": _month_idx(month_yyyymm)})
    m = pd.merge_asof(q.sort_values("거래월_idx"), snap.sort_values("사용가능_idx"), left_on="거래월_idx",
                      right_on="사용가능_idx", direction="backward").sort_values("_pos")
    return pd.DataFrame({"기준월": m["기준월"].to_numpy(), "학원스냅샷_경과개월": (m["거래월_idx"] - m["기준월_idx"]).to_numpy()})


def clusters(store_edu: pd.DataFrame) -> pd.DataFrame:
    base = store_edu[(store_edu["기준월"] == CLUSTER_SNAPSHOT) & store_edu["상권업종소분류코드"].isin(ACADEMY_GROUPS["입시교과"])].copy()
    base["군집"] = DBSCAN(eps=CLUSTER_EPS_M / EARTH_RADIUS_M, min_samples=CLUSTER_MIN_SAMPLES, metric="haversine",
                        algorithm="ball_tree").fit_predict(np.radians(base[["위도", "경도"]].to_numpy()))
    c = (base[base["군집"] >= 0].groupby("군집").agg(학원수=("군집", "size"), 위도=("위도", "mean"), 경도=("경도", "mean"),
                                                   자치구=("시군구명", lambda s: s.mode()[0]), 행정동=("행정동명", lambda s: s.mode()[0]))
         .sort_values("학원수", ascending=False).reset_index(drop=True))
    c["학원가명"] = c["자치구"] + " " + c["행정동"]
    c["3대학원가"] = c["학원수"] >= MAJOR_CLUSTER_SIZE
    return c


def _tree(df):
    return BallTree(np.radians(df[["위도", "경도"]].to_numpy()), metric="haversine")


def compute(query: pd.DataFrame, store_edu: pd.DataFrame, academy_clusters: pd.DataFrame, date_col: str = "계약일자") -> pd.DataFrame:
    """query: 위도 · 경도 · date_col. 반환: query와 같은 인덱스의 학원 Feature 6개 + 스냅샷 메타 · 학원가명."""
    day = pd.to_datetime(query[date_col])
    snap = snapshot_for(day.dt.year * 100 + day.dt.month, normal_snapshots(store_edu))
    if (age := int(snap["학원스냅샷_경과개월"].max())) > MAX_SNAPSHOT_AGE:
        raise ValueError(f"학원 스냅샷이 {age}개월 묵음 — 상가정보 분기 파일 갱신 필요")
    q = pd.DataFrame({"위도": query["위도"].to_numpy(), "경도": query["경도"].to_numpy()})
    q = pd.concat([q, snap], axis=1)
    locs = q[["위도", "경도"]].drop_duplicates().reset_index(drop=True)
    rad = np.radians(locs[["위도", "경도"]].to_numpy())
    parts = []
    for month in sorted(q["기준월"].unique()):
        s, f = store_edu[store_edu["기준월"] == month], locs.copy()
        f["기준월"] = month
        for group, radius, name in RADIUS_RULES:
            f[name] = _tree(s[s["상권업종소분류코드"].isin(ACADEMY_GROUPS[group])]).query_radius(rad, r=radius / EARTH_RADIUS_M, count_only=True)
        parts.append(f)
    static = locs.copy()
    for label, c in [("학원가", academy_clusters), ("3대학원가", academy_clusters[academy_clusters["3대학원가"]])]:
        c = c.reset_index(drop=True)
        dist, idx = _tree(c).query(rad, k=1)
        static[f"최근접{label}_거리_m"] = dist[:, 0] * EARTH_RADIUS_M
        static[f"최근접{label}명"] = c["학원가명"].to_numpy()[idx[:, 0]]
    out = (q.merge(pd.concat(parts, ignore_index=True), on=["위도", "경도", "기준월"], how="left", validate="many_to_one")
           .merge(static, on=["위도", "경도"], how="left", validate="many_to_one").rename(columns={"기준월": "학원스냅샷기준월"}))
    out.index = query.index
    return out[COLUMNS + ["학원스냅샷기준월", "학원스냅샷_경과개월"] + DISPLAY]
