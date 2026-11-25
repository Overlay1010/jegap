"""R14 원천 정정 감지."""

import pandas as pd

from apt_core import vintage
from apt_core.features import market_leading as ml
from conftest import EXTERNAL


def _price():
    return ml.clean_price(ml.read_rone_csv(EXTERNAL / "(월) 매매가격지수_주택종합.csv"), "매매").assign(기준월=lambda d: d["기준월"].astype(str))


def test_same_vintage_has_no_changes():
    p = _price()
    assert vintage.compare(p, p.copy(), ["기준월", "자치구"], ["매매가격지수", "매매가격지수_전월증감률"]).empty


def test_revision_and_new_month_are_handled():
    old = _price()
    new = old.copy()
    hit = (new["기준월"] == "2024-03") & (new["자치구"] == "강남구")
    new.loc[hit, "매매가격지수_전월증감률"] += 0.05                       # 과거 달 정정
    new = pd.concat([new, new.tail(1).assign(기준월="2026-08")])          # 새 달 추가는 정정이 아님
    ch = vintage.compare(old, new, ["기준월", "자치구"], ["매매가격지수", "매매가격지수_전월증감률"])
    assert len(ch) == 1 and ch.iloc[0]["자치구"] == "강남구" and ch.iloc[0]["열"] == "매매가격지수_전월증감률"
    assert vintage.summary(ch)["바뀐_칸"] == 1


def test_rebase_shows_up_as_level_change():
    # 기준시점 변경: 지수 수준이 전부 바뀐다 → 감지돼 알림 대상이 된다 (증감률 · 3개월 비율은 그대로일 수 있음)
    old = _price()
    new = old.assign(매매가격지수=old["매매가격지수"] * 0.9)
    s = vintage.summary(vintage.compare(old, new, ["기준월", "자치구"], ["매매가격지수", "매매가격지수_전월증감률"]))
    assert s["바뀐_칸"] == old["매매가격지수"].notna().sum() and "매매가격지수_전월증감률" not in s["열별_최대차"]
