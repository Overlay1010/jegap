"""정책 Feature 41개 (03_policy 셀 9 ~ 11 · 13 ~ 15 · 20 · 23 ~ 25 이식, 레지스트리 source = 03_policy · 03_policy_trade).

| 묶음 | 단위 | 시점 규칙 |
|---|---|---|
| 지역지정 · 대출 · 세제 상태 14개, 이벤트 집계 12개, 시행 대기 3개, 규제 노출도 3개 | 구 × 기준월 | 발표 < m월 1일 (상태는 시행일 ≤ m월 1일) |
| 계약일 상태 · 토허 동/단지 · 대출 8개 + 예상가격_단지만 | 거래 | 발표 < 계약일 |

입력 표 (사람이 공식 원문으로 관리, 2-2 '반영은 사람 승인'):
- 정책 이벤트 `housing_policy_events_v2.csv`, 구별 · 동별 지정 구간 `policy_zone_status_by_{gu,dong}_v2.csv`
- 토허 '일부(구역 · 단지)' 동의 단지명 규칙 `policy_permit_complex_rules.csv` — 노트북 코드 안 정규식을 표로 옮긴 것.
  pattern이 `{성수 전략정비구역}`이면 구역 경계(geojson) 안 단지 이름으로 바꾼다
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

ZONE_DATES = ["start_date", "end_date", "start_announce_date", "end_announce_date"]
RULE_EVENTS = {
    "P034": "DSR 1단계", "P043": "DSR 2단계", "P044": "DSR 3단계",
    "P098": "스트레스 DSR 1단계", "P114": "스트레스 DSR 2단계", "P128": "스트레스 DSR 3단계",
    "P011": "15억 초과", "P065": "15억 초과", "P064": "LTV 50%", "P141": "LTV 50%→40%", "P130": "6억",
    "P150": "15억 초과~25억", "P151": "25억 초과",
    "P051": "중과 한시 배제", "P069": "연장(1차)", "P101": "연장(2차)", "P119": "연장(3차", "P168": "유예 종료",
}
GRACE_END = pd.Timestamp("2026-05-09")  # P119 종료일, P168로 재연장 없음 확정
STATE_FEATURES = ["투기과열지구", "조정대상지역", "투기지역", "분양가상한제", "분양가상한제_일부동", "토허_구전역", "토허_일부동",
                  "DSR단계", "스트레스DSR단계", "LTV_무주택", "고가주택_주담대금지", "주담대_6억상한", "주담대_가격차등", "양도세중과_적용"]
EVENT_FEATURES = ["정책_강화수_3m", "정책_완화수_3m", "정책_순강도_3m", "정책_강화수_6m", "정책_완화수_6m", "정책_순강도_6m",
                  "대출_순강도_6m", "지역지정_순강도_6m", "세제_순강도_6m", "공급_강도합_6m", "경과개월_최근강화", "경과개월_최근완화"]
PENDING_FEATURES = ["시행대기_강화수", "시행대기_완화수", "시행대기_순강도"]
EXPOSURE_FEATURES = ["평균_한도비율", "상한구속_비중", "고가15억초과_비중"]
REGION_FEATURES = STATE_FEATURES + EVENT_FEATURES + PENDING_FEATURES + EXPOSURE_FEATURES
TRADE_FEATURES = ["투기과열지구_계약일", "조정대상지역_계약일", "토허_구전역_계약일", "양도세중과_계약일", "토허_동노출", "토허_단지지정",
                  "주담대_LTV_적용", "고가주택_주담대금지_해당", "주담대_최대한도_만원", "주담대_한도비율"]
PYEONG, LAG_DAYS, WINDOW_DAYS = 3.3058, 30, 365
SEONGSU_TOKEN = "{성수 전략정비구역}"


def read_events(path) -> pd.DataFrame:
    ev = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["announce_date", "effective_date", "end_date"])
    idx = ev.set_index("event_id")
    for eid, keyword in RULE_EVENTS.items():  # 규칙에 고정한 이벤트가 바뀌면 시작 시 실패
        if keyword not in idx.loc[eid, "policy_name"]:
            raise ValueError(f"규칙 이벤트 {eid} 이름이 바뀜: {idx.loc[eid, 'policy_name']}")
    return ev


def read_zones(path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", parse_dates=ZONE_DATES)


# ---------------- 상태 ----------------
def _rule_active(ev_idx, eid, dates, known_ids=None):
    if known_ids is not None and eid not in known_ids:
        return pd.Series(False, index=dates.index)
    r = ev_idx.loc[eid]
    return (dates > r["announce_date"]) & (dates >= r["effective_date"])


def known_zones(zone_table, known_ids=None):
    # 누수 검증용: 기준 시점에 발표되지 않은 지정은 지우고, 발표되지 않은 해제는 아직 유지 중으로 본다
    if known_ids is None:
        return zone_table
    z = zone_table[zone_table["start_event_id"].isin(known_ids)].copy()
    z.loc[~z["end_event_id"].isin(known_ids), "end_date"] = pd.NaT
    return z


def zone_flags(keys, gu_col, date_col, zone_gu, known_ids=None):
    columns = sorted(zone_gu["designation"].unique())
    m = keys[[gu_col, date_col]].reset_index().merge(known_zones(zone_gu, known_ids), left_on=gu_col, right_on="자치구", how="inner")
    active = ((m["start_announce_date"] < m[date_col]) & (m["start_date"] <= m[date_col])
              & (m["end_date"].isna() | (m["end_date"] >= m[date_col])))
    return (m[active].assign(v=1).pivot_table(index="index", columns="designation", values="v", aggfunc="max")
            .reindex(index=keys.index, columns=columns).fillna(0).astype(int))


def policy_state(keys, gu_col, date_col, events, zone_gu, known_ids=None):
    z = zone_flags(keys, gu_col, date_col, zone_gu, known_ids)
    d, ev_idx = keys[date_col], events.set_index("event_id")
    out = pd.DataFrame(index=keys.index)
    for col, src in [("투기과열지구", "투기과열지구"), ("조정대상지역", "조정대상지역"), ("투기지역", "투기지역"), ("분양가상한제", "분양가상한제"),
                     ("분양가상한제_일부동", "분양가상한제_일부동"), ("토허_구전역", "토허구역_구전역"), ("토허_일부동", "토허구역_일부동")]:
        out[col] = z[src]
    act = lambda eid: _rule_active(ev_idx, eid, d, known_ids)  # noqa: E731
    out["DSR단계"] = act("P034").astype(int) + act("P043") + act("P044")
    out["스트레스DSR단계"] = act("P098").astype(int) + act("P114") + act("P128")
    regulated = out["투기과열지구"].eq(1)
    out["LTV_무주택"] = np.where(regulated, np.select([act("P141"), act("P064")], [40, 50], default=40), 70)
    out["고가주택_주담대금지"] = (regulated & act("P011") & ~act("P065")).astype(int)
    out["주담대_6억상한"] = act("P130").astype(int)
    out["주담대_가격차등"] = (act("P150") & act("P151")).astype(int)
    in_grace = (act("P051") | act("P069") | act("P101") | act("P119")) & (d <= GRACE_END)
    out["양도세중과_적용"] = (out["조정대상지역"].eq(1) & ~in_grace).astype(int)
    return out


def ltv_limit(state, price):
    split_era = state["투기과열지구"].eq(1) & state["고가주택_주담대금지"].eq(1)
    return np.where(split_era, 0.4 * np.minimum(price, 90000) + 0.2 * np.maximum(price - 90000, 0), state["LTV_무주택"] / 100 * price)


def mortgage_limit(state, price):
    """주담대 최대한도(만원, 무주택 실수요자, DSR 미반영)."""
    limit = ltv_limit(state, price)
    limit = np.where(state["투기과열지구"].eq(1) & state["고가주택_주담대금지"].eq(1) & (price > 150000), 0, limit)
    limit = np.where(state["주담대_6억상한"].eq(1), np.minimum(limit, 60000), limit)
    tier_cap = np.select([price <= 150000, price <= 250000], [60000, 40000], default=20000)
    limit = np.where(state["주담대_가격차등"].eq(1), np.minimum(limit, tier_cap), limit)
    return pd.Series(np.where(np.isnan(price), np.nan, limit), index=state.index)


# ---------------- 구 × 기준월 ----------------
def explode_events(ev, all_gu):
    e = ev[ev["event_role"].ne("파생") & ev["target_gu"].notna()].copy()
    e["자치구"] = e["target_gu"].map(lambda s: all_gu if s == "전체" else s.split(";"))
    e = e.explode("자치구")
    e["known_month"] = e["announce_date"].fillna(e["effective_date"]).dt.to_period("M")
    e["is_supply"] = e["policy_category"].eq("공급대책")
    e["sign"] = e["direction"].map({"강화": 1, "완화": -1}).fillna(0).astype(int)
    e["signed"] = np.where(e["is_supply"], 0, e["sign"] * e["intensity"])
    return e


def _dedupe(ev_gu, extra_keys=()):
    # 같은 날 같은 영역 · 같은 방향 발표(한 대책을 여러 행으로 나눈 경우)는 강도가 가장 큰 1건만
    key = ["자치구", "known_date", "policy_group", "is_supply", "sign", *extra_keys]
    return (ev_gu.assign(known_date=ev_gu["announce_date"].fillna(ev_gu["effective_date"]))
            .sort_values(["intensity", "event_id"], ascending=[False, True]).drop_duplicates(key))


def event_aggregates(ev_gu, months_needed, all_gu):
    ev_gu = _dedupe(ev_gu)
    reg, sup = ev_gu[~ev_gu["is_supply"]], ev_gu[ev_gu["is_supply"]]
    months = pd.period_range(min(ev_gu["known_month"].min(), months_needed.min() - 12), months_needed.max(), freq="M")
    panel = pd.MultiIndex.from_product([all_gu, months], names=["자치구", "month"])

    def monthly(df, value):
        return df.groupby(["자치구", "known_month"])[value].sum().rename_axis(["자치구", "month"]).reindex(panel, fill_value=0)

    base = pd.DataFrame({
        "강화수": monthly(reg.assign(v=reg["sign"].eq(1).astype(int)), "v"),
        "완화수": monthly(reg.assign(v=reg["sign"].eq(-1).astype(int)), "v"),
        "순강도": monthly(reg, "signed"),
        "대출": monthly(reg[reg["policy_group"].eq("대출")], "signed"),
        "지역지정": monthly(reg[reg["policy_group"].eq("지역지정")], "signed"),
        "세제": monthly(reg[reg["policy_group"].eq("세제")], "signed"),
        "공급": monthly(sup, "intensity"),
        "강화_major": monthly(reg.assign(v=(reg["sign"].eq(1) & reg["intensity"].ge(2)).astype(int)), "v"),
        "완화_major": monthly(reg.assign(v=(reg["sign"].eq(-1) & reg["intensity"].ge(2)).astype(int)), "v"),
    })
    g = base.groupby(level="자치구", group_keys=False)
    past = lambda col, w: g[col].apply(lambda s: s.rolling(w, min_periods=1).sum().shift(1))  # noqa: E731  m월 = m-w ~ m-1월 발표
    out = pd.DataFrame(index=base.index)
    for name, col, w in [("정책_강화수_3m", "강화수", 3), ("정책_완화수_3m", "완화수", 3), ("정책_순강도_3m", "순강도", 3),
                         ("정책_강화수_6m", "강화수", 6), ("정책_완화수_6m", "완화수", 6), ("정책_순강도_6m", "순강도", 6),
                         ("대출_순강도_6m", "대출", 6), ("지역지정_순강도_6m", "지역지정", 6), ("세제_순강도_6m", "세제", 6),
                         ("공급_강도합_6m", "공급", 6)]:
        out[name] = past(col, w)
    month_no = pd.Series([p.ordinal for p in base.index.get_level_values("month")], index=base.index)
    for col, name in [("강화_major", "경과개월_최근강화"), ("완화_major", "경과개월_최근완화")]:
        last = month_no.where(base[col] > 0).groupby(level="자치구").transform(lambda s: s.ffill().shift(1))
        out[name] = month_no - last
    out = out.reset_index()
    return out[out["month"].isin(months_needed)]


def _pending(ev, g, all_gu):
    e = explode_events(ev, all_gu)
    e = e[e["event_role"].eq("시행") & ~e["is_supply"] & e["sign"].ne(0) & e["effective_date"].notna()]
    e = _dedupe(e, extra_keys=["effective_date"])  # 같은 날 발표라도 시행일이 다르면 따로
    m = g[["자치구", "cutoff"]].reset_index().merge(e[["자치구", "known_date", "effective_date", "sign", "intensity"]], on="자치구")
    m = m[(m["known_date"] < m["cutoff"]) & (m["effective_date"] > m["cutoff"])]
    out = pd.DataFrame(index=g.index)
    out["시행대기_강화수"] = m[m["sign"].eq(1)].groupby("index").size().reindex(g.index, fill_value=0)
    out["시행대기_완화수"] = m[m["sign"].eq(-1)].groupby("index").size().reindex(g.index, fill_value=0)
    out["시행대기_순강도"] = (m["sign"] * m["intensity"]).groupby(m["index"]).sum().reindex(g.index, fill_value=0)
    return out


def reference_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """노출도용 참고 거래: 직거래 제외 (가족 간 · 일괄 매각이 섞여 구 중위보다 약 9% 낮음, 2021-11 이전은 미기록이라 포함)."""
    t = trades[trades["거래유형"].ne("직거래")]
    return pd.DataFrame({"자치구": t["시군구명"].to_numpy(), "ref_month": pd.to_datetime(t["계약일자"]).dt.to_period("M").to_numpy(),
                         "price": t["거래금액"].astype(float).to_numpy()})


def _exposure(g, state, ref):
    # 같은 구 2 ~ 4개월 전 거래(공개 완료된 가격)에 기준월 1일 규제를 적용
    parts = [g[["자치구", "month"]].reset_index().merge(ref.assign(month=ref["ref_month"] + lag), on=["자치구", "month"]) for lag in (2, 3, 4)]
    m = pd.concat(parts, ignore_index=True)
    assert (m["ref_month"] <= m["month"] - 2).all()
    m = m.join(state, on="index")
    price = m["price"].to_numpy()
    limit = mortgage_limit(m, price).to_numpy()
    m["한도비율"], m["상한구속"], m["고가"] = limit / price, (limit < ltv_limit(m, price) - 1).astype(float), (price > 150000).astype(float)
    return (m.groupby("index").agg(평균_한도비율=("한도비율", "mean"), 상한구속_비중=("상한구속", "mean"), 고가15억초과_비중=("고가", "mean"))
            .reindex(g.index))


def region_month(events, grid, zone_gu, ref, known_ids=None):
    """grid: 기준월('YYYY-MM') · 자치구 (+ 권역). 반환: grid + 구 × 월 정책 Feature 32개."""
    all_gu = sorted(grid["자치구"].unique())  # 이벤트 "전체" = grid의 모든 구
    g = grid.copy()
    g["month"] = pd.PeriodIndex(g["기준월"], freq="M")
    g["cutoff"] = g["month"].dt.to_timestamp()
    state = policy_state(g, "자치구", "cutoff", events, zone_gu, known_ids)
    agg = event_aggregates(explode_events(events, all_gu), pd.PeriodIndex(g["month"].unique()), all_gu)
    return pd.concat([g, state, _pending(events, g, all_gu), _exposure(g, state, ref)], axis=1).merge(agg, on=["자치구", "month"], how="left")


# ---------------- 거래 단위 ----------------
def _window_median(df, keys):
    # 각 거래의 같은 key 참고 거래 중 [d-395, d-30) 평당가 중위 · 건수
    day_all, val_all, ref_all = df["day"].to_numpy(), df["평당가"].to_numpy(), df["is_ref"].to_numpy()
    med, cnt = np.full(len(df), np.nan), np.zeros(len(df), dtype=int)
    for idx in df.groupby(keys, observed=True).indices.values():
        r = idx[ref_all[idx]]
        r = r[np.argsort(day_all[r], kind="stable")]
        rd, rv = day_all[r], val_all[r]
        qd = day_all[idx]
        lo = np.searchsorted(rd, qd - (WINDOW_DAYS + LAG_DAYS), side="left")
        hi = np.searchsorted(rd, qd - LAG_DAYS, side="left")
        for j in np.flatnonzero(hi > lo):
            med[idx[j]] = np.median(rv[lo[j]:hi[j]])
            cnt[idx[j]] = hi[j] - lo[j]
    return med, cnt


def expected_price(trades: pd.DataFrame) -> pd.DataFrame:
    """거래 전 예상가격(만원) · 출처. 계약일 30일 전까지 공개됐을 참고 거래(직거래 제외)만, 창 [d-395, d-30).
    우선순위: ① 같은 단지 → ② 같은 법정동 · 면적대 → ③ 같은 구 2개월 전 월 중위."""
    t = trades[["시군구명", "법정동", "단지일련번호", "계약일자", "전용면적", "평당가", "거래유형"]].reset_index(drop=True).copy()
    t["계약일자"] = pd.to_datetime(t["계약일자"])
    t["평당가"] = t["평당가"].astype(float)
    t["면적대"] = pd.cut(t["전용면적"].astype(float), [0, 60, 85, 135, np.inf], labels=["≤60", "60~85", "85~135", ">135"])
    t["day"] = t["계약일자"].to_numpy().astype("datetime64[D]").astype(np.int64)
    t["is_ref"] = t["거래유형"].ne("직거래").to_numpy()
    t["단지_중위평당가"], t["단지_거래수"] = _window_median(t, ["단지일련번호"])
    t["동면적_중위평당가"], t["동면적_거래수"] = _window_median(t, ["시군구명", "법정동", "면적대"])
    gm = (t[t["is_ref"]].groupby(["시군구명", t["계약일자"].dt.to_period("M").rename("month")])["평당가"].median()
          .rename("구_2개월전_중위평당가").reset_index())
    gm["month"] = gm["month"] + 2  # m-2월 값은 m-1월 말까지 모두 신고 → m월 계약 시점에 공개
    t["month"] = t["계약일자"].dt.to_period("M")
    t = t.merge(gm, on=["시군구명", "month"], how="left")
    source = np.select([t["단지_거래수"] > 0, t["동면적_거래수"] > 0, t["구_2개월전_중위평당가"].notna()], ["단지", "동·면적대", "구"], default="없음")
    ppy = np.select([source == "단지", source == "동·면적대", source == "구"],
                    [t["단지_중위평당가"], t["동면적_중위평당가"], t["구_2개월전_중위평당가"]], default=np.nan)
    out = pd.DataFrame({"예상가격_만원": ppy * t["전용면적"].astype(float) / PYEONG, "예상가격_출처": source})
    out["예상가격_단지만"] = out["예상가격_만원"].where(out["예상가격_출처"].eq("단지"))  # 대체값은 신축을 크게 틀려 모델에는 단지 이력만
    out.index = trades.index
    return out


def permit_rules(rules: pd.DataFrame, apt_dong: pd.DataFrame, complexes_xy: pd.DataFrame | None, seongsu_geojson: dict | None) -> pd.DataFrame:
    """단지명 규칙 표 → 동 구간에 붙인다. 성수 행은 구역 경계 안 단지 이름으로 패턴을 만든다 (지정 직후 · 최신 경계 판정이 같아야 함)."""
    rules = rules.copy()
    rules["start_date"] = pd.to_datetime(rules["start_date"])
    if rules["pattern"].eq(SEONGSU_TOKEN).any():
        from shapely.geometry import Point, shape
        from shapely.ops import unary_union
        xy = complexes_xy[complexes_xy["법정동"].isin(["성수동1가", "성수동2가"])]
        inside = {}
        for version in ["2021-05-27", "2026-09-11"]:
            area = unary_union([shape(f["geometry"]) for f in seongsu_geojson["features"] if f["properties"]["버전"] == version])
            inside[version] = set(xy.loc[[area.contains(Point(x, y)) for x, y in zip(xy["경도"], xy["위도"])], "단지일련번호"])
        if inside["2021-05-27"] != inside["2026-09-11"]:
            raise ValueError("성수 전략정비구역 판정이 경계 버전마다 다름")
        if xy.duplicated(["법정동", "단지명"]).any():
            raise ValueError("성수 구역 동에 같은 이름 단지가 있어 단지명 규칙을 쓸 수 없음")
        names = xy[xy["단지일련번호"].isin(inside["2021-05-27"])].groupby("법정동")["단지명"].agg(sorted).to_dict()
        is_s = rules["pattern"].eq(SEONGSU_TOKEN)
        rules.loc[is_s, "pattern"] = rules.loc[is_s, "법정동"].map(lambda d: "^(" + "|".join(map(re.escape, names[d])) + ")$")
    check = rules.merge(apt_dong, on=["자치구", "법정동", "start_date"], how="left", indicator=True)
    if not (check["_merge"].eq("both").all() and check["coverage"].eq("일부(구역·단지)").all()):
        raise ValueError("단지명 규칙이 동별 지정 구간(일부)과 맞지 않음")
    return apt_dong.merge(rules, on=["자치구", "법정동", "start_date"], how="left")


def apartment_permit_zones(zone_dong: pd.DataFrame) -> pd.DataFrame:
    """동별 구간 중 아파트 관련 토허만 (재개발 후보지 · 모아타운 = 비아파트 · 저층주거지 제외)."""
    d = zone_dong[zone_dong["designation"].eq("토지거래허가구역") & ~zone_dong["target_detail"].str.contains("비아파트|저층주거지", na=False)].copy()
    if not d["start_event_id"].notna().all():
        raise ValueError("모든 동 구간은 발표 이벤트와 연결돼야 함 (누수 검증)")
    return d


def permit_zone_flags(tr, apt_dong, date_col="계약일자", known_ids=None):
    m = (tr[["시군구명", "법정동", "단지명", date_col]].reset_index()
         .merge(known_zones(apt_dong, known_ids), left_on=["시군구명", "법정동"], right_on=["자치구", "법정동"], how="inner"))
    m = m[(m["start_announce_date"] < m[date_col]) & (m["start_date"] <= m[date_col])
          & (m["end_date"].isna() | (m["end_date"] >= m[date_col]))].copy()
    whole = m["coverage"].isin(["동전역", "아파트만"])
    by_name = m["pattern"].notna() & m.apply(lambda r: isinstance(r["pattern"], str) and isinstance(r["단지명"], str)
                                             and re.match(r["pattern"], r["단지명"]) is not None, axis=1)
    m["단지적용"] = whole | by_name
    m["근거"] = np.where(whole, "동 " + m["coverage"], m["근거"])
    reason = m[m["단지적용"]].groupby("index")["근거"].agg(lambda s: " / ".join(sorted(set(s.dropna()))))
    out = pd.DataFrame(index=tr.index)
    out["토허_동노출"] = tr.index.isin(m["index"]).astype(int)
    out["토허_단지지정"] = tr.index.isin(m.loc[m["단지적용"], "index"]).astype(int)
    out["토허_근거"] = reason.reindex(tr.index)
    return out


def trade_features(tr, events, zone_gu, apt_dong, known_ids=None):
    """tr: 시군구명 · 법정동 · 단지명 · 계약일자 · 예상가격_만원. 반환: 거래 단위 10개 + 토허_근거."""
    state = policy_state(tr, "시군구명", "계약일자", events, zone_gu, known_ids)
    price = tr["예상가격_만원"].to_numpy(dtype=float)
    out = pd.DataFrame(index=tr.index)
    out["투기과열지구_계약일"] = state["투기과열지구"]
    out["조정대상지역_계약일"] = state["조정대상지역"]
    out["토허_구전역_계약일"] = state["토허_구전역"]
    out["양도세중과_계약일"] = state["양도세중과_적용"]
    out["주담대_LTV_적용"] = state["LTV_무주택"]
    out["고가주택_주담대금지_해당"] = (state["고가주택_주담대금지"].eq(1) & (price > 150000)).astype(int)
    out["주담대_최대한도_만원"] = mortgage_limit(state, price)
    out["주담대_한도비율"] = out["주담대_최대한도_만원"] / price
    permit = permit_zone_flags(tr, apt_dong, known_ids=known_ids)
    gu_wide = out["토허_구전역_계약일"].eq(1)
    out["토허_동노출"] = permit["토허_동노출"]
    out["토허_단지지정"] = (permit["토허_단지지정"].eq(1) | gu_wide).astype(int)  # 구 전역이면 모든 아파트
    out["토허_근거"] = np.where(gu_wide, "구 전역 아파트", permit["토허_근거"])
    out["토허_근거"] = out["토허_근거"].where(out["토허_단지지정"].eq(1))
    return out
