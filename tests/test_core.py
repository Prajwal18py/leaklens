import pandas as pd

from leaklens import LeakLens


def test_full_run_produces_report():
    train = pd.DataFrame({
        "id": range(100),
        "x": range(100),
        "target": [i % 2 for i in range(100)],
    })
    test = train.copy()
    report = LeakLens(train, test, target="target").run()
    assert report.checks_run
    assert "LeakLens Report" in report.summary()


def test_to_json_roundtrip():
    train = pd.DataFrame({"x": range(50), "target": [0, 1] * 25})
    report = LeakLens(train, target="target").run()
    json_str = report.to_json()
    assert "issues" in json_str


def test_to_html_writes_file(tmp_path):
    train = pd.DataFrame({"x": range(50), "y": range(50, 100), "target": [0, 1] * 25})
    test = pd.DataFrame({"x": range(50, 100), "y": range(100, 150), "target": [0, 1] * 25})
    report = LeakLens(train, test, target="target").run()
    out_path = tmp_path / "report.html"
    report.to_html(str(out_path))
    assert out_path.exists()
    assert "LeakLens Report" in out_path.read_text(encoding="utf-8")


def test_invalid_target_raises():
    import pytest
    from leaklens.exceptions import InvalidTargetError
    train = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(InvalidTargetError):
        LeakLens(train, target="nonexistent")
