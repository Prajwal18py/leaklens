import pandas as pd


def to_pandas(df):
    """Convert a polars DataFrame to pandas if needed. Pass pandas DataFrames
    through untouched. This is how LeakLens supports polars without a hard
    dependency on it."""
    if df is None:
        return None
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "to_pandas"):
        try:
            return df.to_pandas()
        except Exception:
            pass
    return df


def detect_datetime_columns(df: pd.DataFrame):
    """Return columns that are already datetime dtype, or that look like
    dates when sampled (so users don't have to pre-parse their date columns)."""
    cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            cols.append(col)
            continue
        if df[col].dtype == object:
            sample = df[col].dropna().head(20)
            if len(sample) == 0:
                continue
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pd.to_datetime(sample, errors="raise")
                cols.append(col)
            except Exception:
                continue
    return cols


def safe_numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include="number").columns.tolist()


def safe_categorical_columns(df: pd.DataFrame):
    return df.select_dtypes(exclude="number").columns.tolist()


def compute_fingerprint(df: pd.DataFrame) -> dict:
    """A reproducible fingerprint of a dataframe: schema hash (column names
    + dtypes) and a distribution hash (rounded summary stats per numeric
    column + value-count signature per categorical column). Lets two runs
    answer 'are we actually looking at the same data?' without comparing
    full datasets — same idea as a file checksum, applied to tabular data.
    """
    import hashlib

    schema_repr = "|".join(f"{c}:{df[c].dtype}" for c in sorted(df.columns))
    schema_hash = hashlib.sha256(schema_repr.encode()).hexdigest()[:16]

    parts = []
    for col in sorted(df.columns):
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                parts.append(f"{col}:{clean.mean():.4g}:{clean.std():.4g}:{clean.min():.4g}:{clean.max():.4g}")
        else:
            top_vals = series.value_counts(normalize=True).head(5)
            parts.append(f"{col}:" + ",".join(f"{v:.4g}" for v in top_vals.values))
    dist_repr = "|".join(parts)
    distribution_hash = hashlib.sha256(dist_repr.encode()).hexdigest()[:16]

    return {
        "schema_hash": schema_hash,
        "distribution_hash": distribution_hash,
        "n_rows": len(df),
        "n_columns": len(df.columns),
    }


def is_likely_identifier_column(series: pd.Series, ratio_threshold: float, absolute_threshold: int) -> bool:
    """Heuristic for 'this is an ID/free-text column, not a real categorical
    feature' — names, ticket numbers, cabin codes, etc. Columns like this
    are near-unique by nature, so distribution-drift and unseen-category
    checks on them produce pure noise rather than real signal.

    Ratio is computed against the non-null count (not total rows), so a
    sparse column like 'Cabin' (mostly missing, but the values it does have
    are almost all unique) is still correctly recognized as identifier-like.
    """
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return False
    nunique = non_null.nunique()
    if nunique < absolute_threshold:
        return False
    return (nunique / n) >= ratio_threshold