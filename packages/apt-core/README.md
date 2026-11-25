# apt-core

V2 노트북에서 검증한 계산을 그대로 옮긴 패키지 (설계: [docs/v4-design.md](../../docs/v4-design.md) 1장).

- **순수 함수:** DataFrame + 기준일을 받아 DataFrame을 돌려준다. 네트워크 · 파일 쓰기 · 환경변수 읽기 없음
- **노트북은 기록:** 노트북은 고치지 않는다. 계산이 바뀌면 여기서만 바꾸고 테스트 기준값도 같이 바꾼다
- **일치 테스트:** 기준값은 커밋된 노트북 산출물(`data/v2_5year_seoul/processed/`, `models/regression/v2/`)

```bash
pip install -e packages/apt-core[test]
pytest packages/apt-core
```

| 모듈 | 노트북 출처 | 테스트 |
|---|---|---|
| `registry.py` | 06 레지스트리 (`v2_feature_registry.csv`) | T8 |
| `features/last_trade.py` | 06 직전 거래 · 시세 보정 (셀 22 · 24) | T1 |
| `features/base.py` | V1 정제(해제만 제외) · 기본 Feature 10개 (V1 01 노트북) | T1 + 경과월 기준 고정 |
| `features/market.py` | 06 구 · 서울 매매 시장 (확정월, 셀 3 · 5) | T1 + V1 재현(시차 1) · 잘라내기 |
| `features/rate.py` | 06 금리 (기준금리 m-1 · 대출금리 m-2, 셀 12) | T1 + 잘라내기 |
| `models/regression.py` | 11 저장 모델 추론 | T4 |
