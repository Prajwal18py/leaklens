import pandas as pd

from leaklens.config import Config
from leaklens.analyzers.schema import SchemaAnalyzer


def test_detects_schema_mismatch():
    train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    test = pd.DataFrame({"a": [1, 2], "c": [3, 4]})
    analyzer = SchemaAnalyzer(Config())
    issues = analyzer.run(train, test)
    assert any(i.title == "Schema Mismatch" for i in issues)


def test_detects_unseen_categories():
    train = pd.DataFrame({"city": ["Delhi", "Mumbai", "Delhi"]})
    test = pd.DataFrame({"city": ["Delhi", "Chennai"]})
    analyzer = SchemaAnalyzer(Config())
    issues = analyzer.run(train, test)
    assert any(i.title == "Unseen Categories" for i in issues)


def test_detects_constant_feature():
    train = pd.DataFrame({"zip": ["560001"] * 10, "x": range(10)})
    analyzer = SchemaAnalyzer(Config())
    issues = analyzer.run(train, None)
    assert any(i.title == "Constant Feature" for i in issues)


def test_detects_dtype_mismatch():
    train = pd.DataFrame({"age": pd.Series([1, 2, 3], dtype="float64")})
    test = pd.DataFrame({"age": pd.Series(["1", "2", "3"], dtype="object")})
    analyzer = SchemaAnalyzer(Config())
    issues = analyzer.run(train, test)
    assert any(i.title == "Dtype Mismatch" for i in issues)
