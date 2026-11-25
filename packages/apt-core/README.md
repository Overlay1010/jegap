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
| `features/complex.py` | 05b 대장 물리 속성 · 공시가격 · 면적 구성 · 재건축 단계 (원본 → 중간 산출물 → 거래 21열) | T1 + 공시 7월 경계 · 미래 고시일 제외. 공시 원본(gitignore)이 없으면 공시 테스트는 건너뜀 |
| `features/school.py` | 02b 학교 23개 (EDSS 학교 표 → 좌표 × 학교기준연도, 6월 공시 경계) | T1 + 학교명 · 기준연도 일치 |
| `features/academy.py` | 02c 학원 6개 (상가정보 교육 업종 → 결손 분기 제외 · 기준월+3개월, 학원가 DBSCAN) | T1 + 학원가 목록 · 스냅샷 일치 |
| `features/market.py` | 06 구 · 서울 매매 시장 (확정월, 셀 3 · 5) | T1 + V1 재현(시차 1) · 잘라내기 |
| `features/rate.py` | 06 금리 (기준금리 m-1 · 대출금리 m-2, 셀 12) | T1 + 잘라내기 |
| `models/regression.py` | 11 저장 모델 추론 | T4 |
