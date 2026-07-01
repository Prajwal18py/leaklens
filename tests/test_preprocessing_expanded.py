import pandas as pd

from leaklens.config import Config
from leaklens.analyzers.preprocessing import PreprocessingLeakageAnalyzer
from leaklens.utils import compute_fingerprint

PCA_LEAKY_SCRIPT = """
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X)
X_train, X_test = train_test_split(X_reduced)
"""

SMOTE_LEAKY_SCRIPT = """
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)
X_train, X_test = train_test_split(X_res, y_res)
"""

CLEAN_SCRIPT = """
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(X)
pca = PCA(n_components=5)
X_train_reduced = pca.fit_transform(X_train)
"""


def test_detects_pca_leak_with_class_name():
    analyzer = PreprocessingLeakageAnalyzer(Config())
    issues = analyzer.analyze_script(PCA_LEAKY_SCRIPT)
    assert any("PCA" in i.message for i in issues)


def test_detects_smote_fit_resample_leak():
    analyzer = PreprocessingLeakageAnalyzer(Config())
    issues = analyzer.analyze_script(SMOTE_LEAKY_SCRIPT)
    assert any("SMOTE" in i.message for i in issues)


def test_clean_script_no_issues():
    analyzer = PreprocessingLeakageAnalyzer(Config())
    issues = analyzer.analyze_script(CLEAN_SCRIPT)
    assert issues == []


def test_fingerprint_is_deterministic():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    fp1 = compute_fingerprint(df)
    fp2 = compute_fingerprint(df)
    assert fp1 == fp2


def test_fingerprint_changes_with_different_data():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"a": [10, 20, 30]})
    fp1 = compute_fingerprint(df1)
    fp2 = compute_fingerprint(df2)
    assert fp1["distribution_hash"] != fp2["distribution_hash"]
    assert fp1["schema_hash"] == fp2["schema_hash"]  # same schema, different values