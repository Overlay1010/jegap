"""원본 대조 테스트 기록 (설계 1-4b R13): '원본을 바꾸거나 파싱 코드를 고치면 원본 대조를 다시 돌린다'를 사람 기억에 맡기지 않는다.
- 원본 대조가 통과하면 그때의 파싱 코드 지문(함수 소스 해시)과 원본 파일 목록(이름 · 크기)을 raw_verified.json에 남긴다
- 평소 테스트(test_raw_verification_is_current, CI 포함)는 지금 코드 · 원본이 기록과 같은지 본다. 다르면 실패하고 돌릴 명령을 알려 준다
- CI에는 원본이 없어 코드 지문만 본다 → 코드만 고치고 기록을 안 올리면 CI에서 잡힌다"""

import hashlib
import inspect
import json
from pathlib import Path

from apt_core.features import academy, school, supply
from conftest import EXTERNAL

RECORD = Path(__file__).with_name("raw_verified.json")
CHECKS = {
    "unsold": {"code": [supply._seoul_rows, supply._to_number, supply._read_month, supply.parse_unsold_month, supply.parse_unsold],
               "raw": lambda: sorted((EXTERNAL / "unsold_raw").glob("unsold_*.xlsx")),
               "run": "APT_CORE_SLOW=1 pytest packages/apt-core -k unsold_parse_all_months"},
    "store": {"code": [academy.extract_education], "consts": [academy.STORE_COLS],
              "raw": lambda: sorted((EXTERNAL / "상가정보").glob("*.csv")),
              "run": "APT_CORE_SLOW=1 pytest packages/apt-core -k store_extract_all_quarters"},
    "edss": {"code": [school._read_edss_csv, school.merge_edss, school._level, school.edss_school_table],
             "consts": [school.EDSS_FILE_GROUPS, school.HIGH_SCHOOL_TYPES, school.BROADCAST_GAP, school.SCHOOL_LOCATION_OVERRIDES],
             "raw": lambda: sorted(EXTERNAL.glob("0[0-9][0-9]*. *(2[1-5]).csv")),
             "run": "pytest packages/apt-core -k edss_raw_to_school_table"},
}


def code_hash(name):
    c = CHECKS[name]
    # 집합은 실행마다 순서가 바뀌므로 정렬해서 지문에 넣는다
    consts = json.dumps(c.get("consts", []), ensure_ascii=False, sort_keys=True, default=lambda x: sorted(x) if isinstance(x, (set, frozenset)) else str(x))
    text = "".join(inspect.getsource(f) for f in c["code"]) + consts
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def raw_list(name):
    return [[p.name, p.stat().st_size] for p in CHECKS[name]["raw"]()]


def load():
    return json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {}


def save(name):
    rec = load()
    rec[name] = {"code": code_hash(name), "raw": raw_list(name)}
    RECORD.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")


def problems():
    rec, out = load(), []
    for name, c in CHECKS.items():
        r = rec.get(name)
        if r is None:
            out.append(f"{name}: 원본 대조 기록 없음 → {c['run']}")
            continue
        if r["code"] != code_hash(name):
            out.append(f"{name}: 파싱 코드가 바뀌었는데 원본 대조를 다시 안 함 → {c['run']} (통과하면 기록이 갱신됨, 같이 커밋)")
        now = raw_list(name)
        if now and now != r["raw"]:  # 원본이 있는 곳(로컬)에서만
            out.append(f"{name}: 원본 파일이 바뀌었는데 원본 대조를 다시 안 함 → {c['run']}")
    return out
