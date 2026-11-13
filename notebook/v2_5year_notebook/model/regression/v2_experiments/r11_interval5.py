# V4 전 점검: 80% 오차 범위가 구간별로도 맞는가 — 등급(적정 / 다소 / 고 · 저평가) 판정의 근거
# - 현재 규칙(models/regression/v2/model_config.json): 예측 월 m마다 m−4 ~ m−2월(확정 3개월) 중개거래 OOF 잔차의 8% · 92% 분위수, 구간 = 단지(기존/신규) × 예측 가격대
# - OOF: models/regression/v2/oof_predictions.parquet (보완 2 모델). Valid = 2025-07~12로 판단, Test = 2026-01~06은 확인만
# [판정 기준 — 결과를 보기 전에 고정, 2026-11-13]
#   점검 구간(기존 단지 · 중개): 세대수(100 미만 / 100~300 / 300~1,000 / 1,000 이상), 직전 같은 면적 거래 경과(180일 이하 / 181~365일 / 365일 초과 · 없음),
#                               연식(35년 이상 여부), 단지 최근 흐름(단지_최근잔차: −3% 미만 / ±3% / +3% 초과, resid_feat.parquet)
#   보완 조건: Valid 거래 1,000건 이상인 구간 중 포함률 < 75%가 하나라도 있으면, 범위 구간에 기준 하나를 더한 후보(세대수 4구간 · 직전 거래 경과 3구간 · 단지 최근 흐름 3구간)를 Valid로 비교
#   채택 조건: Valid 전체 포함률 ≥ 80% 유지 + 점검 구간 중 최저 포함률이 현재보다 올라감 → 그중 Valid 평균 폭이 가장 좁은 것. Test는 채택 뒤 확인만
# 실행: python r11_interval5.py
import os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8'); pd.set_option('display.width', 220)
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), ''); ROOT = os.path.join(S, *['..'] * 5, '')
ALPHA, WIN = 0.08, 3
d = pd.read_parquet(ROOT + 'models/regression/v2/oof_predictions.parquet')
d['월'] = pd.PeriodIndex(d['월'].astype(str), freq='M')
t = pd.read_parquet(ROOT + 'data/v2_5year_seoul/processed/v2_trade.parquet', columns=['transaction_id', '세대수', '직전거래_경과일', '연식'])
rf = pd.read_parquet(S + 'resid_feat.parquet', columns=['transaction_id', '단지_최근잔차'])
d = d.merge(t, on='transaction_id', how='left', validate='one_to_one').merge(rf, on='transaction_id', how='left', validate='one_to_one')
for c in ['세대수', '직전거래_경과일', '연식']: d[c] = pd.to_numeric(d[c], errors='coerce').astype(float)
d['세대수구간'] = pd.cut(d['세대수'], [0, 100, 300, 1000, np.inf], right=False, labels=['100 미만', '100~300', '300~1,000', '1,000 이상']).astype(str)
d['직전경과구간'] = np.select([d['직전거래_경과일'] <= 180, d['직전거래_경과일'] <= 365], ['180일 이하', '181~365일'], '365일 초과 · 없음')
d['연식구간'] = np.where(d['연식'] >= 35, '35년 이상', '35년 미만')
d['흐름구간'] = np.select([d['단지_최근잔차'] < -0.03, d['단지_최근잔차'] > 0.03, d['단지_최근잔차'].notna()], ['−3% 미만', '+3% 초과', '±3%'], '값 없음')
MKT = d['거래유형'].eq('중개거래')
VALM = pd.period_range('2025-07', '2025-12', freq='M'); TESTM = pd.period_range('2026-01', '2026-06', freq='M')
CHECK = ['세대수구간', '직전경과구간', '연식구간', '흐름구간']


def evaluate(months, seg):
    parts = []
    for m in months:
        w = d[MKT & (d['월'] >= m - (WIN + 1)) & (d['월'] <= m - 2)]
        q = w.groupby(seg)['r'].quantile([ALPHA, 1 - ALPHA]).unstack(); qa = w['r'].quantile([ALPHA, 1 - ALPHA])
        sub = d[MKT & (d['월'] == m)].copy(); key = pd.MultiIndex.from_frame(sub[seg]) if len(seg) > 1 else pd.Index(sub[seg[0]])
        sub['lo'] = q[ALPHA].reindex(key).fillna(qa[ALPHA]).to_numpy(); sub['hi'] = q[1 - ALPHA].reindex(key).fillna(qa[1 - ALPHA]).to_numpy()
        parts.append(sub)
    r = pd.concat(parts, ignore_index=True)
    r['포함'] = (r['r'] >= r['lo']) & (r['r'] <= r['hi']); r['폭%'] = (np.exp(r['hi']) - np.exp(r['lo'])) * 100
    return r


def by_segment(r):
    e = r[r['단지'].eq('기존')]; rows = []
    for c in CHECK:
        for k, g in e.groupby(c):
            rows.append({'점검': c, '구간': k, 'n': len(g), '포함률%': g['포함'].mean() * 100, '폭%': g['폭%'].mean(),
                         '아래로 벗어남%': (g['r'] < g['lo']).mean() * 100, '위로 벗어남%': (g['r'] > g['hi']).mean() * 100})
    return pd.DataFrame(rows)


BASE = ['단지', '가격대']
cur_v = evaluate(VALM, BASE)
seg_v = by_segment(cur_v)
print(f"[현재 규칙] Valid 중개 포함률 {cur_v['포함'].mean() * 100:.1f}% · 기존 단지 {cur_v.loc[cur_v['단지'].eq('기존'), '포함'].mean() * 100:.1f}% · 폭 {cur_v['폭%'].mean():.1f}%")
print(seg_v.round(1).to_string(index=False))
big = seg_v[seg_v['n'] >= 1000]
trigger = (big['포함률%'] < 75).any()
print(f"\n보완 조건(1,000건 이상 구간 중 포함률 < 75%): {'해당' if trigger else '해당 없음'} | 1,000건 이상 구간 최저 {big['포함률%'].min():.1f}%")

rows = [{'후보': '현재 (단지 × 가격대)', 'Valid 포함률%': cur_v['포함'].mean() * 100, 'Valid 폭%': cur_v['폭%'].mean(), '점검 구간 최저 포함률%': big['포함률%'].min()}]
results = {'현재 (단지 × 가격대)': (BASE, cur_v)}
if trigger:
    for extra in ['세대수구간', '직전경과구간', '흐름구간']:
        seg = BASE + [extra]; rv = evaluate(VALM, seg); sv = by_segment(rv); sv = sv[sv['n'] >= 1000]
        rows.append({'후보': f'+ {extra}', 'Valid 포함률%': rv['포함'].mean() * 100, 'Valid 폭%': rv['폭%'].mean(), '점검 구간 최저 포함률%': sv['포함률%'].min()})
        results[f'+ {extra}'] = (seg, rv)
cand = pd.DataFrame(rows); print('\n[후보 비교 · Valid]'); print(cand.round(2).to_string(index=False))
ok = cand[(cand.index > 0) & (cand['Valid 포함률%'] >= 80) & (cand['점검 구간 최저 포함률%'] > cand.loc[0, '점검 구간 최저 포함률%'])]
pick = ok.sort_values('Valid 폭%').iloc[0]['후보'] if len(ok) else '현재 (단지 × 가격대)'
print(f"\n→ 채택: {pick}")

seg, _ = results[pick]
for name, s in [('현재 (단지 × 가격대)', BASE)] + ([(pick, seg)] if pick != '현재 (단지 × 가격대)' else []):
    rt = evaluate(TESTM, s); st = by_segment(rt)
    print(f"\n[Test 확인 · {name}] 중개 포함률 {rt['포함'].mean() * 100:.1f}% · 기존 단지 {rt.loc[rt['단지'].eq('기존'), '포함'].mean() * 100:.1f}% · 폭 {rt['폭%'].mean():.1f}%")
    print(st.round(1).to_string(index=False))
out = seg_v.assign(기간='Valid', 규칙='현재'); out.to_csv(S + 'r11_interval5.csv', index=False, encoding='utf-8-sig')
cand.to_csv(S + 'r11_interval5_candidates.csv', index=False, encoding='utf-8-sig')
