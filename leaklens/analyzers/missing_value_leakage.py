from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr

from .base import BaseAnalyzer
from ..models.issue import Issue
from ..models.severity import Severity


class MissingValueLeakageAnalyzer(BaseAnalyzer):
    """Checks whether a column's missingness pattern (whether a value is
    present or absent) is itself correlated with the target — e.g. 'Salary'
    is only missing for fraud cases. This is a real, distinct leakage
    pattern from value-based target leakage: the *fact that data is
    missing* can leak information that the imputed value never would.

    Uses point-biserial correlation for a numeric target and Cramer's V
    (via chi-square) for a categorical target, both computed against a
    binary is-missing indicator — same statistical machinery already used
    elsewhere in the library, just applied to missingness instead of value.
    """

    name = "missing_value_leakage"

    def run(self, train, test=None, target: Optional[str] = None) -> List[Issue]:
        issues: List[Issue] = []
        if target is None or target not in train.columns:
            return issues

        y = train[target]
        is_numeric_target = pd.api.types.is_numeric_dtype(y)

        for col in train.columns:
            if col == target:
                continue
            missing_mask = train[col].isna()
            if missing_mask.sum() == 0 or missing_mask.sum() == len(train):
                continue  # no variation in missingness — nothing to correlate

            if is_numeric_target:
                try:
                    corr, p_value = pointbiserialr(missing_mask.astype(int), y)
                except Exception:
                    continue
                if pd.isna(corr):
                    continue
                if abs(corr) >= self.config.missingness_corr_threshold and p_value < 0.05:
                    issues.append(Issue(
                        title="Missing Value Leakage",
                        severity=Severity.WARNING,
                        column=col,
                        analyzer=self.name,
                        message=(
                            f"Whether '{col}' is missing correlates with the target "
                            f"(point-biserial r={corr:.3f}, p={p_value:.4f}) — the absence "
                            f"of a value may itself be leaking information."
                        ),
                        details={"missingness_correlation": float(corr), "p_value": float(p_value)},
                    ))
            else:
                try:
                    confusion = pd.crosstab(missing_mask, y)
                    if confusion.shape[0] < 2 or confusion.shape[1] < 2:
                        continue
                    chi2 = chi2_contingency(confusion)[0]
                    n = confusion.sum().sum()
                    if n <= 1:
                        continue
                    phi2 = chi2 / n
                    r, k = confusion.shape
                    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
                    rcorr = r - ((r - 1) ** 2) / (n - 1)
                    kcorr = k - ((k - 1) ** 2) / (n - 1)
                    denom = min(kcorr - 1, rcorr - 1)
                    if denom <= 0:
                        continue
                    v = float(np.sqrt(phi2corr / denom))
                except Exception:
                    continue
                if v >= self.config.missingness_corr_threshold:
                    issues.append(Issue(
                        title="Missing Value Leakage",
                        severity=Severity.WARNING,
                        column=col,
                        analyzer=self.name,
                        message=(
                            f"Whether '{col}' is missing correlates with the target "
                            f"(Cramer's V={v:.3f}) — the absence of a value may itself "
                            f"be leaking information."
                        ),
                        details={"missingness_cramers_v": v},
                    ))
        return issues