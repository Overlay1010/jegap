"""
현재 목록만 주는 원천의 월 스냅숏 (v4-design 2-2 `snapshot-monthly`, 5-10 — 서비스 공개 전에는 개발 PC에서 매달 실행)

이 원천들은 과거 시점 값을 다시 받을 수 없다 (V2 한계: "현재 목록이라 과거 시점 복원 불가").
지금부터 매달 받아 두면 이후 기간은 시점을 반영할 수 있다.

| 단계 | 원천 | 방법 |
|---|---|---|
| subway | 지하철역 출입구 (서울교통빅데이터플랫폼) | 06c 실행 → 결과를 스냅숏으로 옮기고, 추적 중인 V2 입력 파일은 원래대로 되돌린다 |
| kapt | K-apt 단지 목록 · 기본 · 상세 (data.go.kr 15057332 · 15058453) | 05c 실행 (raw 캐시는 달마다 새로) |
| permit | 주택인허가 기본개요 (data.go.kr 15136560) | 04a 실행 (raw 캐시는 달마다 새로) |
| building | 건축물대장 표제부 · 총괄표제부 (건축HUB) | 04e title · recap 실행. 일일 호출 한도에 걸리면 다음 날 다시 실행하면 이어서 받는다 |
| manual | 대규모점포 · 정비사업 추진현황(서울시) · 학구도(반기) — 받는 곳은 아래 MANUAL_SOURCES | 원본 파일을 data/snapshots/_inbox/<원천>/ 에 두면 날짜를 붙여 보관한다 |

- 저장: data/snapshots/<원천>/<파일명>_YYYYMMDD.<확장자>, 목록 data/snapshots/_manifest.csv (원천 · 파일 · sha256 · 바이트 · 저장일)
- 직전 스냅숏과 내용이 같으면(수집일 열 제외, subway) 저장하지 않고 목록에 "같음"만 남긴다
- raw 캐시: 처음 실행할 때 V2 때 받은 캐시는 data/snapshots/_raw_v2/ 로 옮겨 보관하고, 이후 달이 바뀌면 지난달 캐시를 지운다
- data/snapshots/ 는 Git에 올리지 않는다 (.gitignore)

실행 (프로젝트 루트에서):
    python notebook/v2_5year_notebook/eda/v2/07_snapshot_monthly.py all        # subway · kapt · permit · building · manual
    python notebook/v2_5year_notebook/eda/v2/07_snapshot_monthly.py subway     # 한 단계만
    python notebook/v2_5year_notebook/eda/v2/07_snapshot_monthly.py status     # 원천별 마지막 스냅숏
"""

import csv
import hashlib
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path


def find_project_root(start=Path(__file__).resolve()):
    for path in [start, *start.parents]:
        if (path / "data").exists() and (path / ".env").exists():
            return path
    raise FileNotFoundError("프로젝트 루트를 찾을 수 없습니다.")


PROJECT_DIR = find_project_root()
HERE = Path(__file__).resolve().parent
EXTERNAL_DIR = PROJECT_DIR / "data" / "external"
SNAP_DIR = PROJECT_DIR / "data" / "snapshots"
MANIFEST = SNAP_DIR / "_manifest.csv"
TODAY = date.today()
STAMP, MONTH = TODAY.strftime("%Y%m%d"), TODAY.strftime("%Y-%m")
MANUAL_SOURCES = {  # 받는 곳 확인 2026-11-25. 다운로드 버튼이 페이지 안 스크립트라 데이터셋 페이지에서 받는다
    "large_store": "대규모점포 인허가 CSV — https://www.data.go.kr/data/15045013/fileData.do (행정안전부_생활_대규모점포, 수시 갱신, 좌표 EPSG:5174)",
    "jeongbi": "서울시 정비사업 추진현황 XLSX — https://data.seoul.go.kr/dataList/OA-22856/F/1/datasetView.do (분기, 새 기준월 파일만)",
    "schoolzone": "학구도 (반기, 3월 · 9월) — 초등학교통학구역 SHP https://www.data.go.kr/data/15159265/fileData.do · 학교학구도연계정보 CSV https://www.data.go.kr/data/15159266/fileData.do · 중학교학구 SHP (공공데이터포털에서 '한국교육시설안전원 중학교학구' 검색)",
}
# 유치원 목록은 스냅숏하지 않는다: 유치원알리미 공시 자료가 공시차수별(2013년 1차 ~)로 남아 있어 과거 시점을 다시 받을 수 있다
# (https://e-childschoolinfo.moe.go.kr/openData.do). V2의 "폐원 유치원 복원 불가" 한계를 이 자료로 풀 수 있는지는 별도 과제


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_rows():
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def record(source, path, digest, note):
    new = not MANIFEST.exists()
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["원천", "파일", "sha256", "바이트", "저장일", "비고"])
        size = path.stat().st_size if path and path.exists() else ""
        w.writerow([source, path.relative_to(PROJECT_DIR).as_posix() if path else "", digest, size, TODAY.isoformat(), note])


def last_digest(source, key=""):
    rows = [r for r in manifest_rows() if r["원천"] == source and key in r["파일"] and r["sha256"]]
    return rows[-1]["sha256"] if rows else None


def keep(source, src, move=True, compare_digest=None):
    """src 파일을 data/snapshots/<source>/에 날짜를 붙여 보관. 직전 스냅숏과 같으면 저장하지 않는다."""
    digest = compare_digest or sha256(src)
    stem = src.stem.removesuffix(f"_{STAMP}")
    if digest == last_digest(source, stem):
        record(source, None, digest, f"{src.name}: 직전 스냅숏과 같음 → 저장 안 함")
        if move:
            src.unlink()
        print(f"[{source}] {src.name}: 직전과 같음")
        return
    out = SNAP_DIR / source / f"{stem}_{STAMP}{src.suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)
    (shutil.move if move else shutil.copy2)(src, out)
    record(source, out, digest, "")
    print(f"[{source}] 저장 {out.relative_to(PROJECT_DIR)}")


def run(script, *args):
    print(f"$ python {script} {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, str(HERE / script), *args], cwd=PROJECT_DIR)
    if r.returncode != 0:
        raise SystemExit(f"{script} 실패 (종료 코드 {r.returncode}) — 원인을 확인하고 이 단계를 다시 실행하세요")


def fresh_raw(raw_dir, name):
    """raw 캐시를 이번 달 것으로 맞춘다. 같은 달이면 그대로(이어 받기), 달이 바뀌었으면 비운다."""
    marker = raw_dir / "_snapshot_month.txt"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == MONTH:
        return
    if raw_dir.exists() and any(raw_dir.iterdir()):
        if marker.exists():
            shutil.rmtree(raw_dir)
        else:  # V2 때 받은 캐시: 지우지 않고 한 번 옮겨 보관
            dest = SNAP_DIR / "_raw_v2" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(raw_dir), str(dest))
            print(f"V2 raw 캐시 보관: {dest.relative_to(PROJECT_DIR)}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(MONTH, encoding="utf-8")


def step_subway():
    target = EXTERNAL_DIR / "subway_entrances.csv"  # Git이 추적하는 V2 입력 → 실행 뒤 원래 내용으로 되돌린다
    original = target.read_bytes()
    try:
        run("06c_download_subway_entrances.py")
        text = target.read_text(encoding="utf-8-sig")
        body = "\n".join(",".join(line.split(",")[:-1]) for line in text.splitlines())  # 마지막 열 = 수집일
        tmp = SNAP_DIR / "_tmp" / "subway_entrances.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(target.read_bytes())
        keep("subway", tmp, compare_digest=hashlib.sha256(body.encode("utf-8")).hexdigest())
    finally:
        target.write_bytes(original)


def step_kapt():
    fresh_raw(EXTERNAL_DIR / "kapt" / "raw", "kapt_raw")
    run("05c_collect_kapt.py")
    for kind in ("list", "basic", "detail"):
        keep("kapt", EXTERNAL_DIR / "kapt" / f"kapt_{kind}_seoul_{STAMP}.parquet")


def step_permit():
    fresh_raw(EXTERNAL_DIR / "housing_permit_raw", "housing_permit_raw")
    run("04a_collect_housing_permit.py")
    keep("permit", EXTERNAL_DIR / f"housing_permit_basis_seoul_{STAMP}.parquet")


def step_building():
    fresh_raw(EXTERNAL_DIR / "building_register_raw", "building_register_raw")
    fresh_raw(EXTERNAL_DIR / "building_register_recap_raw", "building_register_recap_raw")
    for kind in ("title", "recap"):
        run("04e_collect_building_register.py", kind)
        keep("building", EXTERNAL_DIR / f"building_register_{kind}_seoul_{STAMP}.parquet")


def step_manual():
    inbox = SNAP_DIR / "_inbox"
    found = False
    for source, desc in MANUAL_SOURCES.items():
        d = inbox / source
        d.mkdir(parents=True, exist_ok=True)
        files = [p for p in d.iterdir() if p.is_file()]
        for p in files:
            keep(source, p)
            found = True
        if not files:
            print(f"[{source}] 받을 것: {desc} → {d.relative_to(PROJECT_DIR)}/ 에 두고 'manual'을 다시 실행")
    if not found:
        print("수동 원천은 이번 달 아직 없음 (위 안내)")


def step_status():
    rows = manifest_rows()
    if not rows:
        print("스냅숏 없음")
        return
    last = {}
    for r in rows:
        last[r["원천"]] = r
    for source, r in last.items():
        print(f"{source:13s} {r['저장일']}  {r['파일'] or r['비고']}")


STEPS = {"subway": step_subway, "kapt": step_kapt, "permit": step_permit, "building": step_building, "manual": step_manual, "status": step_status}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "status"
    if arg == "all":
        for name in ("subway", "kapt", "permit", "building", "manual"):
            print(f"\n===== {name} =====", flush=True)
            STEPS[name]()
    elif arg in STEPS:
        STEPS[arg]()
    else:
        sys.exit(f"사용법: 07_snapshot_monthly.py [all | {' | '.join(STEPS)}]")
