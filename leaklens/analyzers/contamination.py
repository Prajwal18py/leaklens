from typing import List, Optional
import pandas as pd

from .base import BaseAnalyzer
from ..models.issue import Issue
from ..models.severity import Severity


class ContaminationAnalyzer(BaseAnalyzer):
    """Catches the classic 'oops I sampled both sets from the same pool
    without removing overlap' bug, plus plain duplicate rows within train."""

    name = "contamination"

    def run(self, train, test=None, target: Optional[str] = None) -> List[Issue]:
        issues: List[Issue] = []

        dup_train = int(train.duplicated().sum())
        if dup_train > 0:
            pct_dup = dup_train / len(train) * 100
            issues.append(Issue(
                title="Duplicate Rows",
                severity=Severity.WARNING if pct_dup > 1 else Severity.INFO,
                analyzer=self.name,
                message=f"{dup_train} duplicate rows found in train set ({pct_dup:.2f}%).",
                details={"duplicate_count": dup_train, "duplicate_pct": pct_dup},
            ))

        if test is None:
            return issues

        common_cols = [c for c in train.columns if c in test.columns]
        if not common_cols:
            return issues

        train_hashes = set(self._hash_rows(train[common_cols]))
        test_hash_series = self._hash_rows(test[common_cols])
        n_overlap = sum(1 for h in test_hash_series if h in train_hashes)

        if n_overlap > 0:
            pct = n_overlap / len(test_hash_series) * 100 if len(test_hash_series) else 0
            severity = (
                Severity.CRITICAL if pct >= self.config.contamination_critical_pct
                else Severity.WARNING if pct >= self.config.contamination_warning_pct
                else Severity.INFO
            )
            issues.append(Issue(
                title="Train/Test Contamination",
                severity=severity,
                analyzer=self.name,
                message=(
                    f"{n_overlap} test rows ({pct:.2f}%) are identical to rows in train — "
                    f"possible split leakage."
                ),
                details={"overlap_count": int(n_overlap), "overlap_pct": float(pct)},
            ))
        return issues

    @staticmethod
    def _hash_rows(df: pd.DataFrame):
        return pd.util.hash_pandas_object(df.astype(str), index=False).tolist()
