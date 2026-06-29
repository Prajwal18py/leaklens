import pandas as pd

from leaklens.config import Config
from leaklens.analyzers.temporal import TemporalLeakageAnalyzer


def test_detects_temporal_overlap():
    train = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=100, freq="D")})
    test = pd.DataFrame({"date": pd.date_range("2023-02-01", periods=20, freq="D")})  # overlaps train range
    analyzer = TemporalLeakageAnalyzer(Config())
    issues = analyzer.run(train, test)
    assert any(i.title == "Temporal Leakage" for i in issues)


def test_no_overlap_when_chronological():
    train = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=100, freq="D")})
    test = pd.DataFrame({"date": pd.date_range("2023-05-01", periods=20, freq="D")})
    analyzer = TemporalLeakageAnalyzer(Config())
    issues = analyzer.run(train, test)
    assert not issues
