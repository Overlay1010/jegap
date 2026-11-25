"""T8 레지스트리 계약: 모델 입력 = 레지스트리 role == model 201개, 금지 열 없음."""

from apt_core import registry as reg


def test_model_columns_match_saved_model(registry, model_config):
    cols = reg.model_columns(registry)
    assert len(cols) == 201
    assert set(cols) == set(model_config["features"])
    assert len(model_config["features"]) == len(set(model_config["features"]))


def test_no_forbidden_column_is_model_input(registry, model_config):
    assert not (set(reg.model_columns(registry)) & reg.FORBIDDEN)
    assert not (set(model_config["features"]) & reg.FORBIDDEN)


def test_v2_trade_has_all_model_columns(v2_trade, model_config):
    assert set(model_config["features"]) <= set(v2_trade.columns)
    assert len(v2_trade) == 234_302
