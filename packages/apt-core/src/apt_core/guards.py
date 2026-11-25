"""서빙 행 검사 (설계 1-4b R11): 월 단위 입력(금리 · 시장 · 공급 · 인구 · 정책 등)이 갱신되지 않으면 그 Feature가 결측인 채로
모델이 예측한다. 학습 데이터 최근 12개월에 한 번도 결측이 없던 열은 서빙 행에서도 결측이면 안 된다 → 한 함수로 모두 막는다.
단지마다 원래 없을 수 있는 열(직전 거래 · 대장 · 공시 연결 실패 등)은 학습 최근 구간에도 결측이 있어 대상에서 빠진다."""

from __future__ import annotations

import pandas as pd


def required_columns(train: pd.DataFrame, features: list[str], month_col: str = "기준월", recent_months: int = 12) -> list[str]:
    months = sorted(train[month_col].astype(str).unique())[-recent_months:]
    recent = train[train[month_col].astype(str).isin(months)]
    return [c for c in features if not recent[c].isna().any()]


def check_required(rows: pd.DataFrame, required: list[str]) -> None:
    bad = rows[required].isna().sum()
    if (bad := bad[bad > 0]).any():
        raise ValueError(f"학습 때 결측이 없던 Feature가 비어 있음 (입력 갱신 확인): {bad.to_dict()}")
