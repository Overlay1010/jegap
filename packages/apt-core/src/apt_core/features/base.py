"""V1 기본 정제 · 기본 Feature 10개 (V1 01_base_eda_feature_engineering 셀 9 · 15 · 42 · 60 이식, 레지스트리 source = V1).

정제: 해제 거래(해제여부 == "O")만 뺀다. 거래 고유 ID가 없어 같은 조건 반복 거래도 지우지 않는다 (V1 처리 정책).
      transaction_id = 해제를 뺀 원본 행 순서 (V2 학습 표 234,302행과 같음).
Feature: 전용면적 · 층 · 연식 · 경과월 · 월_sin · 월_cos · 위도 · 경도 · 거래유형 · 토지임대부아파트여부.
- 연식 = 계약년도 − 건축년도, 0 미만 · 100 초과는 결측
- 경과월 = 2021-07부터 센 개월 수. V1은 '학습 데이터의 첫 달'로 셌는데, 서빙에서는 데이터 범위가 바뀌므로 BASE_MONTH로 고정한다
- 위도 · 경도 = 지번 주소("서울특별시 구 법정동 지번") → 지오코딩 캐시 (카카오 지번 검색 + 실패 시 도로명 검색 결과).
  새 주소의 지오코딩은 pipeline이 캐시에 채운다 (apt-core는 네트워크를 쓰지 않음)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_MONTH = pd.Timestamp("2021-07-01")
COLUMNS = ["전용면적", "층", "연식", "경과월", "월_sin", "월_cos", "위도", "경도", "거래유형", "토지임대부아파트여부"]

SEOUL_GU = {
    "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구", "11215": "광진구",
    "11230": "동대문구", "11260": "중랑구", "11290": "성북구", "11305": "강북구", "11320": "도봉구",
    "11350": "노원구", "11380": "은평구", "11410": "서대문구", "11440": "마포구", "11470": "양천구",
    "11500": "강서구", "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
    "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구", "11740": "강동구",
}


def clean(trades_raw: pd.DataFrame) -> pd.DataFrame:
    """국토부 매매 원본 → 해제 제외 거래 표 (transaction_id · 시군구명 · 계약일자 · 거래월일자 추가)."""
    t = trades_raw[trades_raw["해제여부"].ne("O")].reset_index(drop=True).copy()
    t.insert(0, "transaction_id", np.arange(len(t)))
    t["법정동시군구코드"] = t["법정동시군구코드"].astype(str).str.zfill(5)
    t["시군구명"] = t["법정동시군구코드"].map(SEOUL_GU)
    if t["시군구명"].isna().any():
        raise ValueError(f"서울 밖 시군구 코드: {sorted(t.loc[t['시군구명'].isna(), '법정동시군구코드'].unique())[:5]}")
    t["단지일련번호"] = t["단지일련번호"].astype(str)
    t["계약일자"] = pd.to_datetime(dict(year=t["계약년도"], month=t["계약월"], day=t["계약일"]))
    t["거래월일자"] = t["계약일자"].values.astype("datetime64[M]").astype("datetime64[ns]")
    return t


def search_address(rows: pd.DataFrame) -> pd.Series:
    """지오코딩 키: "서울특별시 {구} {법정동} {지번}" (V1 셀 42)."""
    parts = [rows[c].astype("string").str.strip().fillna("") for c in ("시군구명", "법정동", "지번")]
    return ("서울특별시 " + parts[0] + " " + parts[1] + " " + parts[2]).str.replace(r"\s+", " ", regex=True).str.strip()


def compute(rows: pd.DataFrame, geocode_cache: pd.DataFrame) -> pd.DataFrame:
    """rows: clean() 출력 형식(계약일자 · 건축년도 · 시군구명 · 법정동 · 지번 · 전용면적 · 층 · 거래유형 · 토지임대부아파트여부).
    반환: rows와 같은 인덱스의 기본 Feature 10개."""
    out = pd.DataFrame(index=rows.index)
    day = pd.to_datetime(rows["계약일자"])
    year = day.dt.year.astype("Int64")
    month = day.dt.month.astype("Int64")
    out["전용면적"] = rows["전용면적"].astype(float)
    out["층"] = rows["층"].astype("int64")
    age = year - pd.to_numeric(rows["건축년도"], errors="coerce").astype("Int64")
    out["연식"] = age.mask((age < 0) | (age > 100))
    out["경과월"] = ((day.dt.year - BASE_MONTH.year) * 12 + day.dt.month - BASE_MONTH.month).astype("int32")
    out["월_sin"] = np.sin(2 * np.pi * month / 12)
    out["월_cos"] = np.cos(2 * np.pi * month / 12)
    geo = (geocode_cache[["검색주소", "위도", "경도"]].drop_duplicates("검색주소", keep="last").set_index("검색주소"))
    key = search_address(rows)
    out["위도"] = key.map(geo["위도"]).astype(float).to_numpy()
    out["경도"] = key.map(geo["경도"]).astype(float).to_numpy()
    # 원본 API는 거래유형이 없던 시기(2021 초) 값을 문자열 "nan"으로 준다 → 결측으로 (V1과 같음)
    out["거래유형"] = rows["거래유형"].where(~rows["거래유형"].astype(str).isin(["nan", "", "None"]), None)
    out["토지임대부아파트여부"] = rows["토지임대부아파트여부"]
    return out[COLUMNS]
