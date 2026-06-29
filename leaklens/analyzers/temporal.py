from typing import List, Optional
import pandas as pd

from .base import BaseAnalyzer
from ..models.issue import Issue
from ..models.severity import Severity
from ..utils import detect_datetime_columns


class TemporalLeakageAnalyzer(BaseAnalyzer):
    """If a date/timestamp column exists, checks whether the test set
    contains rows that occur before the latest date in train — i.e. the
    split isn't actually chronological when it probably should be."""

    name = "temporal_leakage"

    def run(self, train, test=None, target: Optional[str] = None) -> List[Issue]:
        issues: List[Issue] = []
        if test is None:
            return issues

        date_cols = [c for c in detect_datetime_columns(train) if c in test.columns]

        for col in date_cols:
            try:
                train_dates = pd.to_datetime(train[col], errors="coerce").dropna()
                test_dates = pd.to_datetime(test[col], errors="coerce").dropna()
            except Exception:
                continue
            if train_dates.empty or test_dates.empty:
                continue

            train_max = train_dates.max()
            overlap_count = int((test_dates < train_max).sum())

            if overlap_count > 0:
                pct = overlap_count / len(test_dates) * 100
                issues.append(Issue(
                    title="Temporal Leakage",
                    severity=Severity.CRITICAL if pct > 5 else Severity.WARNING,
                    column=col,
                    analyzer=self.name,
                    message=(
                        f"{overlap_count} test rows ({pct:.1f}%) have '{col}' dates earlier "
                        f"than the latest train date — split may not be chronological."
                    ),
                    details={"overlap_count": overlap_count, "overlap_pct": float(pct)},
                ))
        return issues
