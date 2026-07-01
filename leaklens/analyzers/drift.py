from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .base import BaseAnalyzer
from ..models.issue import Issue
from ..models.severity import Severity
from ..utils import safe_numeric_columns, is_likely_identifier_column


class DriftAnalyzer(BaseAnalyzer):
    """Compares train vs test distributions column by column: KS test for
    numeric columns, Population Stability Index (PSI) for categorical ones.

    Identifier-like categorical columns (names, ticket numbers, free-text
    IDs) are skipped — every value differing between train/test is expected
    for a near-unique column, so PSI/unseen-category checks there are noise,
    not signal. Numeric columns are never skipped this way, since KS drift
    on a numeric ID-like column (e.g. PassengerId) is still meaningful.
    """

    name = "drift"

    def run(self, train, test=None, target: Optional[str] = None) -> List[Issue]:
        issues: List[Issue] = []
        if test is None:
            return issues

        numeric_cols = set(safe_numeric_columns(train))
        common_cols = [c for c in train.columns if c in test.columns and c != target]

        for col in common_cols:
            if col in numeric_cols and col in safe_numeric_columns(test):
                issues.extend(self._check_numeric_drift(col, train[col], test[col]))
            else:
                if is_likely_identifier_column(
                    train[col],
                    self.config.high_cardinality_ratio_threshold,
                    self.config.high_cardinality_absolute_threshold,
                ):
                    continue
                issues.extend(self._check_categorical_drift(col, train[col], test[col]))
        return issues

    def _check_numeric_drift(self, col, train_series, test_series) -> List[Issue]:
        train_clean = train_series.dropna()
        test_clean = test_series.dropna()
        if len(train_clean) < 5 or len(test_clean) < 5:
            return []
        stat, p_value = ks_2samp(train_clean, test_clean)
        if p_value < self.config.ks_alpha:
            severity = Severity.CRITICAL if p_value < 0.001 else Severity.WARNING
            return [Issue(
                title="Distribution Drift",
                severity=severity,
                column=col,
                analyzer=self.name,
                message=f"KS test p-value={p_value:.4f} — train/test distributions differ significantly.",
                details={
                    "ks_stat": float(stat), "p_value": float(p_value), "test": "ks",
                    **self._root_cause(train_clean, test_clean),
                },
            )]
        return []

    @staticmethod
    def _root_cause(train_clean, test_clean) -> dict:
        """Plain statistics explaining *what shape* of drift occurred — not
        a model, not an LLM, just comparing summary statistics between the
        two samples so the report can say more than 'drift detected'."""
        causes = []
        train_mean, test_mean = train_clean.mean(), test_clean.mean()
        train_std, test_std = train_clean.std(), test_clean.std()
        train_p95, test_p95 = train_clean.quantile(0.95), test_clean.quantile(0.95)

        if train_std > 0 and abs(test_mean - train_mean) / train_std > 0.3:
            direction = "increased" if test_mean > train_mean else "decreased"
            causes.append(f"Mean {direction} ({train_mean:.3g} → {test_mean:.3g})")
        if train_std > 0 and abs(test_std - train_std) / train_std > 0.3:
            direction = "increased" if test_std > train_std else "decreased"
            causes.append(f"Variance {direction} (std {train_std:.3g} → {test_std:.3g})")
        if train_p95 != 0 and abs(test_p95 - train_p95) / abs(train_p95) > 0.3:
            causes.append(f"95th percentile shifted ({train_p95:.3g} → {test_p95:.3g})")

        return {"root_cause": causes} if causes else {}

    def _check_categorical_drift(self, col, train_series, test_series) -> List[Issue]:
        psi = self._calculate_psi(train_series, test_series)
        if psi is None:
            return []
        if psi >= self.config.psi_critical:
            severity = Severity.CRITICAL
        elif psi >= self.config.psi_warning:
            severity = Severity.WARNING
        else:
            return []
        return [Issue(
            title="Distribution Drift",
            severity=severity,
            column=col,
            analyzer=self.name,
            message=f"PSI={psi:.3f} — category distribution shifted between train and test.",
            details={"psi": float(psi), "test": "psi"},
        )]

    @staticmethod
    def _calculate_psi(train_series, test_series, eps: float = 1e-6) -> Optional[float]:
        train_dist = train_series.value_counts(normalize=True)
        test_dist = test_series.value_counts(normalize=True)
        categories = set(train_dist.index) | set(test_dist.index)
        if not categories:
            return None
        psi = 0.0
        for cat in categories:
            p = max(train_dist.get(cat, eps), eps)
            q = max(test_dist.get(cat, eps), eps)
            psi += (p - q) * np.log(p / q)
        return abs(float(psi))