import pandas as pd

from leaklens.config import Config
from leaklens.analyzers.contamination import ContaminationAnalyzer


def test_detects_overlap_rows():
    train = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"]})
    test = pd.DataFrame({"a": [1, 2, 99], "b": ["x", "y", "q"]})  # rows 0,1 duplicated from train
    analyzer = ContaminationAnalyzer(Config())
    issues = analyzer.run(train, test)
    assert any(i.title == "Train/Test Contamination" for i in issues)


def test_detects_duplicate_rows_in_train():
    train = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
    analyzer = ContaminationAnalyzer(Config())
    issues = analyzer.run(train, None)
    assert any(i.title == "Duplicate Rows" for i in issues)


def test_clean_split_has_no_contamination():
    train = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"]})
    test = pd.DataFrame({"a": [5, 6, 7], "b": ["p", "q", "r"]})
    analyzer = ContaminationAnalyzer(Config())
    issues = analyzer.run(train, test)
    assert not any(i.title == "Train/Test Contamination" for i in issues)
