import numpy as np
import pandas as pd

from leaklens.config import Config
from leaklens.analyzers.target_leakage import TargetLeakageAnalyzer
from leaklens.models.severity import Severity


def test_detects_categorical_leakage():
    df = pd.DataFrame({
        "leaky_cat": (["A", "A", "A", "B", "B", "B"]) * 5,
        "target": ([1, 1, 1, 0, 0, 0]) * 5,
    })
    analyzer = TargetLeakageAnalyzer(Config())
    issues = analyzer.run(df, target="target")
    assert any(i.column == "leaky_cat" and i.severity == Severity.CRITICAL for i in issues)


def test_detects_high_correlation_numeric_leakage():
    rng = np.random.default_rng(0)
    target = rng.normal(size=300)
    df = pd.DataFrame({
        "leaky_num": target * 1.0001,
        "target": target,
    })
    analyzer = TargetLeakageAnalyzer(Config())
    issues = analyzer.run(df, target="target")
    assert any(i.column == "leaky_num" for i in issues)


def test_no_leakage_on_random_data():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "feature": rng.normal(size=200),
        "target": rng.integers(0, 2, size=200),
    })
    analyzer = TargetLeakageAnalyzer(Config())
    issues = analyzer.run(df, target="target")
    assert not any(i.severity == Severity.CRITICAL for i in issues)


def test_returns_empty_without_target():
    df = pd.DataFrame({"a": [1, 2, 3]})
    analyzer = TargetLeakageAnalyzer(Config())
    assert analyzer.run(df, target=None) == []
