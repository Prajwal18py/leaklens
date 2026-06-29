import numpy as np
import pandas as pd

from leaklens.config import Config
from leaklens.analyzers.drift import DriftAnalyzer
from leaklens.models.severity import Severity


def test_detects_numeric_drift():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    test = pd.DataFrame({"x": rng.normal(5, 1, 500)})  # shifted distribution
    analyzer = DriftAnalyzer(Config())
    issues = analyzer.run(train, test)
    assert any(i.column == "x" for i in issues)


def test_no_drift_on_same_distribution():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    test = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    analyzer = DriftAnalyzer(Config())
    issues = analyzer.run(train, test)
    assert not any(i.severity == Severity.CRITICAL for i in issues)


def test_detects_categorical_drift_via_psi():
    train = pd.DataFrame({"city": (["Delhi"] * 90 + ["Mumbai"] * 10)})
    test = pd.DataFrame({"city": (["Delhi"] * 10 + ["Mumbai"] * 90)})
    analyzer = DriftAnalyzer(Config())
    issues = analyzer.run(train, test)
    assert any(i.column == "city" for i in issues)
