from dataclasses import dataclass


@dataclass
class Config:
    """Tunable thresholds for every analyzer. Pass a custom Config to LeakLens
    if your dataset needs looser/stricter checks than the defaults."""

    # Target leakage
    target_corr_threshold: float = 0.95
    cramers_v_threshold: float = 0.90

    # Drift
    ks_alpha: float = 0.05
    psi_warning: float = 0.10
    psi_critical: float = 0.25

    # Contamination
    contamination_warning_pct: float = 1.0
    contamination_critical_pct: float = 5.0

    # Duplicate columns
    duplicate_column_corr_threshold: float = 0.999

    # Constant / near-constant features
    constant_feature_threshold: float = 0.99

    # Columns above this cardinality ratio (unique values / row count) are
    # treated as identifiers (names, ticket numbers, free-text IDs) rather
    # than real categorical features, and are skipped by the categorical
    # drift (PSI) and unseen-categories checks — both produce pure noise on
    # near-unique columns, since every value differing between train/test is
    # expected, not a data quality issue.
    high_cardinality_ratio_threshold: float = 0.3
    high_cardinality_absolute_threshold: int = 50

    # Missing value leakage — correlation between a column's missingness
    # and the target (point-biserial for numeric target, Cramer's V for
    # categorical target)
    missingness_corr_threshold: float = 0.3