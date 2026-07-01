import ast
from typing import List

from .base import BaseAnalyzer
from ..models.issue import Issue
from ..models.severity import Severity


class PreprocessingLeakageAnalyzer(BaseAnalyzer):
    """Inspects a script (source code or a .py file path) for common
    real-world leakage bugs: fitting a transformer, imputer, encoder,
    feature selector, or oversampler (StandardScaler, PCA, KNNImputer,
    LabelEncoder, TargetEncoder, SMOTE, SelectKBest, etc.) BEFORE
    train_test_split() is called.

    This analyzer is not dataframe-driven, so it isn't part of the standard
    .run() pipeline — call analyze_script() directly, or pass `script=` to
    LeakLens() and it will be wired in automatically.
    """

    name = "preprocessing_leakage"

    # Class names commonly fit on data that must not see the test set
    # before a split — used to make the finding message more specific.
    KNOWN_LEAKY_CLASSES = {
        "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer",
        "PCA", "TruncatedSVD", "KernelPCA",
        "KNNImputer", "SimpleImputer", "IterativeImputer",
        "LabelEncoder", "OneHotEncoder", "OrdinalEncoder", "TargetEncoder",
        "SMOTE", "ADASYN", "RandomOverSampler", "RandomUnderSampler",
        "SelectKBest", "SelectPercentile", "RFE", "VarianceThreshold",
    }

    def run(self, train, test=None, target=None) -> List[Issue]:
        return []

    def analyze_script(self, source: str) -> List[Issue]:
        issues: List[Issue] = []
        code = source
        if source.strip().endswith(".py"):
            try:
                with open(source, "r") as f:
                    code = f.read()
            except OSError:
                return issues

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return issues

        # Track which variable names were instantiated from a known-leaky
        # class, so a bare `imputer.fit(X)` can be traced back to e.g. KNNImputer.
        var_classes = self._track_variable_classes(tree)

        split_line = None
        fit_calls = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name == "train_test_split" and split_line is None:
                    split_line = node.lineno
                elif func_name in ("fit", "fit_transform", "fit_resample"):
                    caller = self._get_caller_name(node)
                    fit_calls.append((node.lineno, caller, var_classes.get(caller)))

        if split_line is not None:
            for lineno, caller, cls_name in fit_calls:
                if lineno < split_line:
                    label = f"{cls_name} ('{caller}')" if cls_name else caller
                    issues.append(Issue(
                        title="Preprocessing-Before-Split Leakage",
                        severity=Severity.CRITICAL,
                        analyzer=self.name,
                        message=(
                            f"Line {lineno}: {label}.fit(...) is called before "
                            f"train_test_split() on line {split_line} — this leaks test "
                            f"data into preprocessing statistics."
                        ),
                        details={"fit_line": lineno, "split_line": split_line, "caller": caller, "class": cls_name},
                    ))
        return issues

    def _track_variable_classes(self, tree: ast.AST) -> dict:
        """Maps variable name -> class name for assignments like
        `scaler = StandardScaler()`, so later `scaler.fit(X)` calls can be
        attributed to a specific known-leaky transformer class."""
        mapping = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                call_name = self._get_call_name(node.value)
                if call_name in self.KNOWN_LEAKY_CLASSES:
                    for target_node in node.targets:
                        if isinstance(target_node, ast.Name):
                            mapping[target_node.id] = call_name
        return mapping

    @staticmethod
    def _get_call_name(node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    @staticmethod
    def _get_caller_name(node: ast.Call) -> str:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            return node.func.value.id
        return "object"