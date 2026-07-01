import numpy as np
import pandas as pd

from leaklens.config import Config
from leaklens.analyzers.missing_value_leakage import MissingValueLeakageAnalyzer


def test_detects_missingness_correlated_with_numeric_target():
    rng = np.random.default_rng(0)
    n = 200
    target = rng.integers(0, 2, n)
    # 'salary' is missing exactly when target == 1
    salary = rng.normal(50000, 5000, n)
    salary = pd.Series(salary)
    salary[target == 1] = np.nan
    df = pd.DataFrame({"salary": salary, "target": target})

    analyzer = MissingValueLeakageAnalyzer(Config())
    issues = analyzer.run(df, target="target")
    assert any(i.column == "salary" for i in issues)


def test_no_leakage_when_missingness_is_random():
    rng = np.random.default_rng(1)
    n = 300
    target = rng.integers(0, 2, n)
    col = rng.normal(size=n)
    mask = rng.random(n) < 0.2  # random missingness, unrelated to target
    col = pd.Series(col)
    col[mask] = np.nan
    df = pd.DataFrame({"feature": col, "target": target})

    analyzer = MissingValueLeakageAnalyzer(Config())
    issues = analyzer.run(df, target="target")
    assert not any(i.column == "feature" for i in issues)


def test_returns_empty_without_target():
    df = pd.DataFrame({"a": [1, None, 3]})
    analyzer = MissingValueLeakageAnalyzer(Config())
    assert analyzer.run(df, target=None) == []


def test_skips_columns_with_no_missing_values():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    analyzer = MissingValueLeakageAnalyzer(Config())
    issues = analyzer.run(df, target="target")
    assert issues == []