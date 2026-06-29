from typing import List, Optional
import numpy as np
import pandas as pd

from .base import BaseAnalyzer
from ..models.issue import Issue
from ..models.severity import Severity


class TargetLeakageAnalyzer(BaseAnalyzer):
    """Flags features that look like they leak the target:
    - a feature whose values map almost 1:1 to a single target value
      (e.g. an ID column that secretly encodes the label)
    - numeric features with suspiciously high correlation to a numeric target
    - categorical features with high Cramer's V against a categorical target
    """

    name = "target_leakage"

    def run(self, train, test=None, target: Optional[str] = None) -> List[Issue]:
        issues: List[Issue] = []
        if target is None or target not in train.columns:
            return issues

        y = train[target]
        is_numeric_target = pd.api.types.is_numeric_dtype(y)

        for col in train.columns:
            if col == target:
                continue
            series = train[col]

            if self._is_near_identical_mapping(train, col, target):
                grouped = train.groupby(col)[target].nunique()
                mapped_fraction = (grouped == 1).mean()
                issues.append(Issue(
                    title="Target Leakage",
                    severity=Severity.CRITICAL,
                    column=col,
                    analyzer=self.name,
                    message=(
                        f"{mapped_fraction * 100:.1f}% of '{col}' values map to a single "
                        f"target value — this likely leaks the target."
                    ),
                    details={"mapped_fraction": float(mapped_fraction)},
                ))
                continue

            if is_numeric_target and pd.api.types.is_numeric_dtype(series):
                corr = self._safe_corr(series, y)
                if corr is not None and abs(corr) >= self.config.target_corr_threshold:
                    issues.append(Issue(
                        title="Target Leakage",
                        severity=Severity.CRITICAL if abs(corr) > 0.99 else Severity.WARNING,
                        column=col,
                        analyzer=self.name,
                        message=f"Correlation with target is {corr:.3f} — investigate for leakage.",
                        details={"correlation": float(corr)},
                    ))
            elif not is_numeric_target and series.dtype == object:
                v = self._cramers_v(series, y)
                if v is not None and v >= self.config.cramers_v_threshold:
                    issues.append(Issue(
                        title="Target Leakage",
                        severity=Severity.WARNING,
                        column=col,
                        analyzer=self.name,
                        message=f"Cramer's V with target is {v:.3f} — strong association, check for leakage.",
                        details={"cramers_v": float(v)},
                    ))
        return issues

    @staticmethod
    def _is_near_identical_mapping(train: pd.DataFrame, col: str, target: str) -> bool:
        try:
            n_unique = train[col].nunique()
            n_rows = len(train)
            if n_unique <= 1:
                return False
            # If almost every row has its own unique value (e.g. a continuous
            # numeric feature or a near-ID column), each "group" trivially
            # contains one row and trivially maps to one target value. That's
            # not leakage, it's just high cardinality — require real grouping
            # (multiple rows per value on average) before treating this as a
            # leakage signal.
            avg_group_size = n_rows / n_unique
            if avg_group_size < 2:
                return False
            grouped = train.groupby(col)[target].nunique()
            if len(grouped) < 2:
                return False
            return bool((grouped == 1).mean() > 0.98)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _safe_corr(a: pd.Series, b: pd.Series) -> Optional[float]:
        try:
            corr = a.corr(b)
            return float(corr) if pd.notna(corr) else None
        except Exception:
            return None

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> Optional[float]:
        try:
            from scipy.stats import chi2_contingency
            confusion = pd.crosstab(x, y)
            if confusion.size == 0:
                return None
            chi2 = chi2_contingency(confusion)[0]
            n = confusion.sum().sum()
            if n <= 1:
                return None
            phi2 = chi2 / n
            r, k = confusion.shape
            phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            denom = min(kcorr - 1, rcorr - 1)
            if denom <= 0:
                return None
            return float(np.sqrt(phi2corr / denom))
        except Exception:
            return None
