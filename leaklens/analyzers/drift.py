import numpy as np
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import gaussian_kde


def numeric_drift_figure(col: str, train_series: pd.Series, test_series: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=train_series.dropna(), name="Train", opacity=0.6, histnorm="probability density",
        marker_color="#6366f1",
    ))
    fig.add_trace(go.Histogram(
        x=test_series.dropna(), name="Test", opacity=0.6, histnorm="probability density",
        marker_color="#f97316",
    ))
    fig.update_layout(
        barmode="overlay",
        title=f"Distribution Drift — {col}",
        xaxis_title=col,
        yaxis_title="Density",
        template="plotly_white",
        height=350,
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig


def numeric_drift_kde_figure(col: str, train_series: pd.Series, test_series: pd.Series):
    """Smooth KDE overlay — reads better than overlapping histogram bars,
    especially when train/test have different sample sizes or bin alignment
    makes the histogram look noisy."""
    train_clean = train_series.dropna().astype(float)
    test_clean = test_series.dropna().astype(float)

    fig = go.Figure()
    if len(train_clean) >= 2 and train_clean.nunique() > 1:
        x_min = min(train_clean.min(), test_clean.min()) if len(test_clean) else train_clean.min()
        x_max = max(train_clean.max(), test_clean.max()) if len(test_clean) else train_clean.max()
        x_grid = np.linspace(x_min, x_max, 200)

        train_kde = gaussian_kde(train_clean)
        fig.add_trace(go.Scatter(
            x=x_grid, y=train_kde(x_grid), name="Train", mode="lines",
            line=dict(color="#6366f1", width=3), fill="tozeroy",
            fillcolor="rgba(99,102,241,0.15)",
        ))
        if len(test_clean) >= 2 and test_clean.nunique() > 1:
            test_kde = gaussian_kde(test_clean)
            fig.add_trace(go.Scatter(
                x=x_grid, y=test_kde(x_grid), name="Test", mode="lines",
                line=dict(color="#f97316", width=3), fill="tozeroy",
                fillcolor="rgba(249,115,22,0.15)",
            ))
    fig.update_layout(
        title=f"Distribution Drift — {col}",
        xaxis_title=col,
        yaxis_title="Density",
        template="plotly_white",
        height=320,
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig


def categorical_drift_figure(col: str, train_series: pd.Series, test_series: pd.Series):
    train_dist = train_series.value_counts(normalize=True)
    test_dist = test_series.value_counts(normalize=True)
    categories = sorted(set(train_dist.index) | set(test_dist.index), key=str)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(c) for c in categories], y=[train_dist.get(c, 0) for c in categories],
        name="Train", marker_color="#6366f1",
    ))
    fig.add_trace(go.Bar(
        x=[str(c) for c in categories], y=[test_dist.get(c, 0) for c in categories],
        name="Test", marker_color="#f97316",
    ))
    fig.update_layout(
        barmode="group",
        title=f"Category Distribution — {col}",
        yaxis_title="Proportion",
        template="plotly_white",
        height=350,
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig