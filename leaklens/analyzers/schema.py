from typing import List, Optional
import pandas as pd

from .base import BaseAnalyzer
from ..models.issue import Issue
from ..models.severity import Severity


class SchemaAnalyzer(BaseAnalyzer):
    """Bundles the structural checks: constant/near-constant features,
    schema mismatch between train/test, dtype mismatch, and unseen
    categories in test that weren't present in train (which breaks many
    encoders at inference time)."""

    name = "schema"

    def run(self, train, test=None, target: Optional[str] = None) -> List[Issue]:
        issues: List[Issue] = []
        issues.extend(self._check_constant_features(train))
        if test is not None:
            issues.extend(self._check_schema_mismatch(train, test, target))
            issues.extend(self._check_dtype_mismatch(train, test))
            issues.extend(self._check_unseen_categories(train, test, target))
        return issues

    def _check_constant_features(self, train: pd.DataFrame) -> List[Issue]:
        issues = []
        for col in train.columns:
            nunique = train[col].nunique(dropna=False)
            if nunique <= 1:
                issues.append(Issue(
                    title="Constant Feature",
                    severity=Severity.WARNING,
                    column=col,
                    analyzer=self.name,
                    message=f"'{col}' has only {nunique} unique value(s) — provides no information.",
                    details={"nunique": int(nunique)},
                ))
            else:
                top_freq = train[col].value_counts(normalize=True, dropna=False).iloc[0]
                if top_freq >= self.config.constant_feature_threshold:
                    issues.append(Issue(
                        title="Near-Constant Feature",
                        severity=Severity.INFO,
                        column=col,
                        analyzer=self.name,
                        message=f"'{col}' is {top_freq * 100:.1f}% a single value — very low variance.",
                        details={"top_value_freq": float(top_freq)},
                    ))
        return issues

    def _check_schema_mismatch(self, train, test, target: Optional[str]) -> List[Issue]:
        train_cols = set(train.columns) - ({target} if target else set())
        test_cols = set(test.columns) - ({target} if target else set())
        only_train = train_cols - test_cols
        only_test = test_cols - train_cols
        if only_train or only_test:
            return [Issue(
                title="Schema Mismatch",
                severity=Severity.CRITICAL,
                analyzer=self.name,
                message=(
                    f"Train and test have different columns. "
                    f"Only in train: {sorted(only_train) or 'none'}. "
                    f"Only in test: {sorted(only_test) or 'none'}."
                ),
                details={"only_train": sorted(only_train), "only_test": sorted(only_test)},
            )]
        return []

    def _check_dtype_mismatch(self, train, test) -> List[Issue]:
        issues = []
        common = [c for c in train.columns if c in test.columns]
        for col in common:
            if str(train[col].dtype) != str(test[col].dtype):
                issues.append(Issue(
                    title="Dtype Mismatch",
                    severity=Severity.WARNING,
                    column=col,
                    analyzer=self.name,
                    message=f"'{col}' is {train[col].dtype} in train but {test[col].dtype} in test.",
                    details={"train_dtype": str(train[col].dtype), "test_dtype": str(test[col].dtype)},
                ))
        return issues

    def _check_unseen_categories(self, train, test, target: Optional[str]) -> List[Issue]:
        issues = []
        cat_cols = [
            c for c in train.select_dtypes(exclude="number").columns
            if c in test.columns and c != target
        ]
        for col in cat_cols:
            train_vals = set(train[col].dropna().unique())
            test_vals = set(test[col].dropna().unique())
            unseen = test_vals - train_vals
            if unseen:
                pct = len(unseen) / max(len(test_vals), 1) * 100
                preview = sorted(str(v) for v in unseen)[:5]
                suffix = "..." if len(unseen) > 5 else ""
                issues.append(Issue(
                    title="Unseen Categories",
                    severity=Severity.WARNING,
                    column=col,
                    analyzer=self.name,
                    message=f"{len(unseen)} categories in test not seen in train: {preview}{suffix}",
                    details={"unseen_count": len(unseen), "unseen_pct": float(pct)},
                ))
        return issues
