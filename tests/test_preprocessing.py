from leaklens.config import Config
from leaklens.analyzers.preprocessing import PreprocessingLeakageAnalyzer

LEAKY_SCRIPT = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)
"""

CLEAN_SCRIPT = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
"""


def test_detects_leaky_pattern():
    analyzer = PreprocessingLeakageAnalyzer(Config())
    issues = analyzer.analyze_script(LEAKY_SCRIPT)
    assert any(i.title == "Preprocessing-Before-Split Leakage" for i in issues)


def test_clean_pattern_has_no_issue():
    analyzer = PreprocessingLeakageAnalyzer(Config())
    issues = analyzer.analyze_script(CLEAN_SCRIPT)
    assert issues == []
