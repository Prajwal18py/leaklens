import ast
from typing import List

from .base import BaseAnalyzer
from ..models.issue import Issue
from ..models.severity import Severity


class PreprocessingLeakageAnalyzer(BaseAnalyzer):
    """Inspects a script (source code or a .py file path) for the most
    common real-world leakage bug: fitting a transformer (StandardScaler,
    OneHotEncoder, SimpleImputer, etc.) on the full dataset BEFORE
    train_test_split() is called.

    This analyzer is not dataframe-driven, so it isn't part of the standard
    .run() pipeline — call analyze_script() directly, or pass `script=` to
    LeakLens() and it will be wired in automatically.
    """

    name = "preprocessing_leakage"

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

        split_line = None
        fit_calls = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name == "train_test_split" and split_line is None:
                    split_line = node.lineno
                elif func_name in ("fit", "fit_transform"):
                    fit_calls.append((node.lineno, self._get_caller_name(node)))

        if split_line is not None:
            for lineno, caller in fit_calls:
                if lineno < split_line:
                    issues.append(Issue(
                        title="Preprocessing-Before-Split Leakage",
                        severity=Severity.CRITICAL,
                        analyzer=self.name,
                        message=(
                            f"Line {lineno}: '{caller}.fit(...)' is called before "
                            f"train_test_split() on line {split_line} — this leaks test "
                            f"data into preprocessing statistics."
                        ),
                        details={"fit_line": lineno, "split_line": split_line, "caller": caller},
                    ))
        return issues

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
