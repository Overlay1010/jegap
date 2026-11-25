"""R13: 원본 대조를 다시 돌려야 하는데 안 돌렸으면 평소 테스트가 실패한다 (raw_record.py 참고)."""

import raw_record


def test_raw_verification_is_current():
    assert raw_record.problems() == []


def test_changed_code_is_caught(monkeypatch):
    monkeypatch.setattr(raw_record, "code_hash", lambda name: "바뀐코드")
    assert all("파싱 코드가 바뀌었는데" in p for p in raw_record.problems())
