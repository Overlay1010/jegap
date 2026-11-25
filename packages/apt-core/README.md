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
| `models/regression.py` | 11 저장 모델 추론 | T4 |
