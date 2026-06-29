import plotly.graph_objects as go
import pandas as pd


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
