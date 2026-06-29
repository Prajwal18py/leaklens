from itertools import combinations
from typing import List, Optional
import pandas as pd

from .base import BaseAnalyzer
from ..models.issue import Issue
from ..models.severity import Severity


class DuplicateColumnsAnalyzer(BaseAnalyzer):
    """Catches the 'price' vs 'selling_price' problem: near-perfectly
    correlated numeric columns, or categorical columns with identical values."""

    name = "duplicate_columns"

    def run(self, train, test=None, target: Optional[str] = None) -> List[Issue]:
        issues: List[Issue] = []

        numeric_cols = train.select_dtypes(include="number").columns.tolist()
        for col_a, col_b in combinations(numeric_cols, 2):
            try:
                corr = train[col_a].corr(train[col_b])
            except Exception:
                continue
            if pd.notna(corr) and abs(corr) >= self.config.duplicate_column_corr_threshold:
                issues.append(Issue(
                    title="Duplicate Columns",
                    severity=Severity.WARNING,
                    analyzer=self.name,
                    message=f"'{col_a}' and '{col_b}' are nearly identical (corr={corr:.5f}) — consider dropping one.",
                    details={"column_a": col_a, "column_b": col_b, "correlation": float(corr)},
                ))

        non_numeric_cols = train.select_dtypes(exclude="number").columns.tolist()
        for col_a, col_b in combinations(non_numeric_cols, 2):
            try:
                if train[col_a].equals(train[col_b]):
                    issues.append(Issue(
                        title="Duplicate Columns",
                        severity=Severity.WARNING,
                        analyzer=self.name,
                        message=f"'{col_a}' and '{col_b}' contain identical values.",
                        details={"column_a": col_a, "column_b": col_b},
                    ))
            except Exception:
                continue
        return issues
