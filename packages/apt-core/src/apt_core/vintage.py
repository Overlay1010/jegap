"""원천 정정 감지 (설계 1-4b R14): 공식 통계는 과거 값을 정정할 수 있다 (부동산원 기준시점 변경 때 과거 지수 재계산, KOSIS 사후 수정 등).
매달 재학습이 최신 원본으로 Feature를 다시 만들므로 학습 · 서빙은 일관되지만, 정정이 조용히 일어나면 모른다 →
새로 받은 원본을 직전 보관본과 겹치는 기간에서 비교해 바뀐 칸을 돌려준다. pipeline이 결과를 pipeline_runs에 남기고, 기준을 넘으면 알림.
공식 정정은 정상이므로 실패시키지 않는다."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compare(old: pd.DataFrame, new: pd.DataFrame, keys: list[str], cols: list[str], tol: float = 0.0) -> pd.DataFrame:
    """겹치는 키에서 값이 바뀐 칸 (키 · 열 · 이전 · 새 값). 이전 본에 있었는데 새 본에서 사라진 키는 열 = '<행 삭제>'."""
    o, n = old.set_index(keys)[cols], new.set_index(keys)[cols]
    if o.index.duplicated().any() or n.index.duplicated().any():
        raise ValueError("키가 중복됨")
    both = o.index.intersection(n.index)
    a, b = o.loc[both], n.loc[both]
    rows = []
    for c in cols:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        changed = ~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= tol))
        for k, v0, v1 in zip(both[changed], x[changed], y[changed]):
            rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "열": c, "이전": v0, "새": v1})
    for k in o.index.difference(n.index):
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "열": "<행 삭제>", "이전": np.nan, "새": np.nan})
    return pd.DataFrame(rows, columns=keys + ["열", "이전", "새"])


def summary(changes: pd.DataFrame) -> dict:
    """pipeline_runs 기록용: 바뀐 칸 수 · 열별 최대 절대 차이."""
    v = changes[changes["열"] != "<행 삭제>"]
    return {"바뀐_칸": int(len(v)), "삭제_행": int((changes["열"] == "<행 삭제>").sum()),
            "열별_최대차": {c: float(g["새"].sub(g["이전"]).abs().max()) for c, g in v.groupby("열")}}
