import pandas as pd

from leaklens.config import Config
from leaklens.analyzers.duplicate_columns import DuplicateColumnsAnalyzer


def test_detects_duplicate_numeric_columns():
    df = pd.DataFrame({"price": [10, 20, 30, 40], "selling_price": [10, 20, 30, 40]})
    analyzer = DuplicateColumnsAnalyzer(Config())
    issues = analyzer.run(df, None)
    assert any(i.title == "Duplicate Columns" for i in issues)


def test_detects_duplicate_categorical_columns():
    df = pd.DataFrame({"city": ["A", "B", "C"], "city_copy": ["A", "B", "C"]})
    analyzer = DuplicateColumnsAnalyzer(Config())
    issues = analyzer.run(df, None)
    assert any(i.title == "Duplicate Columns" for i in issues)


def test_no_false_positive_on_unrelated_columns():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 1, 8, 2]})
    analyzer = DuplicateColumnsAnalyzer(Config())
    issues = analyzer.run(df, None)
    assert issues == []
