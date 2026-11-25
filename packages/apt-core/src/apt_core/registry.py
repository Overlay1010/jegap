"""Feature 레지스트리 = 모듈 출력의 계약 (06 노트북 '레지스트리' 셀이 만든 v2_feature_registry.csv).

- role == "model" 인 열만 모델 입력이다 (V2 최종 201개)
- 금지 열(정답 · 거래 이후 정보)은 어떤 모듈도 모델 입력으로 내보내면 안 된다
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# 06 노트북 FORBIDDEN 그대로
FORBIDDEN = frozenset({
    "거래금액", "log_price", "평당가", "log_평당가", "향후3개월_변화율", "상승여부", "공식_향후3개월_변화율", "상승세", "강한상승",
    "등기일자", "해제여부", "해제사유발생일", "매도자", "매수자", "중개사소재지",
})


def load(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def model_columns(registry: pd.DataFrame, dataset: str = "v2_trade") -> list[str]:
    r = registry[registry["dataset"].eq(dataset) & registry["role"].eq("model")]
    return r["column"].tolist()


def columns_by_source(registry: pd.DataFrame, source: str, dataset: str = "v2_trade") -> list[str]:
    r = registry[registry["dataset"].eq(dataset) & registry["role"].eq("model") & registry["source"].eq(source)]
    return r["column"].tolist()
