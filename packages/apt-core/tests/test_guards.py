"""R11 서빙 행 결측 검사."""

import json

import numpy as np
import pytest

from apt_core import guards
from conftest import MODEL_DIR


@pytest.fixture(scope="module")
def saved():
    return json.load(open(MODEL_DIR / "required_non_null.json", encoding="utf-8"))["columns"]


def test_saved_list_matches_training(v2_trade, model_config, saved):
    assert guards.required_columns(v2_trade, model_config["features"]) == saved
    assert len(saved) == 165


def test_recent_rows_pass(v2_trade, saved):
    guards.check_required(v2_trade[v2_trade["기준월"] >= "2025-07"], saved)


def test_stale_monthly_input_fails(v2_trade, saved):
    rows = v2_trade[v2_trade["기준월"] == "2026-06"].head(3).copy()
    rows["기준금리_전월"] = np.nan  # 금리 표가 갱신되지 않은 달
    with pytest.raises(ValueError, match="기준금리_전월"):
        guards.check_required(rows, saved)
