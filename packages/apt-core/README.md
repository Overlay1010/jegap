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
| `features/policy.py` | 03 정책 41개 (구 × 월 32 · 거래 9, 예상가격 포함). 토허 단지명 정규식은 `data/external/policy_permit_complex_rules.csv`로 옮김 | T1 + 기준월 · 계약일 이후 발표 제외 · 잘못 넣은 구간을 잡는 시험의 시험 · `check_zone_links`(이벤트 ↔ 구간 표 연결, 만료됐는데 열린 구간은 **오늘 날짜로** 실패) |
| `features/supply.py` | 04 공급 14개(공식 인허가 · 준공, 미분양, 청약홈 분양공고) + 06 재고 대비 6개 | T1 + m-2월 · 공고일 이후 제외 · 시차 1이면 실패 |
| `features/demographics.py` | 05 인구 15개 + 구별 재고 (KOSIS 원본 표 7개부터) | T1 + 데이터월 m-2 이후 제외 · 시차 1이면 실패 |
| `features/market_leading.py` | 01 시장 선행 16개 (R-ONE 수급 · 가격지수 CSV, 한국은행 CSI · CCSI 엑셀 원본부터) | T1 + 공표 전 값 제외 · 시차 줄이면 실패 · 월이 빠지면 실패 |
| `features/market.py` | 06 구 · 서울 매매 시장 (확정월, 셀 3 · 5) | T1 + V1 재현(시차 1) · 잘라내기 |
| `features/rate.py` | 06 금리 (기준금리 m-1 · 대출금리 m-2, 셀 12) | T1 + 잘라내기 |
| `models/regression.py` | 11 저장 모델 추론 | T4 |
| `guards.py` | 서빙 행 검사: 학습 최근 12개월 결측 없던 165개 열 (`required_non_null.json`) | R11 |

원본 → 중간 산출물 단계도 옮겼다: 미분양 엑셀(`supply.parse_unsold`) · EDSS(`school.merge_edss` · `edss_school_table`) · 상가정보(`academy.extract_education`). 원본은 gitignore라 CI에서는 건너뛰고, 전체 대조는 `APT_CORE_SLOW=1 pytest packages/apt-core`. 대조가 통과하면 `tests/raw_verified.json`에 코드 지문 · 원본 목록이 기록되고, 원본이나 파싱 코드가 바뀌었는데 다시 안 돌리면 평소 테스트가 실패한다 (기록 파일도 같이 커밋).
