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
