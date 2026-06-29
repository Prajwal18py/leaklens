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