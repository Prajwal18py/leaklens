import plotly.graph_objects as go


def contamination_summary_figure(overlap_pct: float, duplicate_pct: float):
    fig = go.Figure(go.Bar(
        x=["Train/Test Overlap", "Duplicate Rows (Train)"],
        y=[overlap_pct, duplicate_pct],
        marker_color=["#ef4444", "#f59e0b"],
        text=[f"{overlap_pct:.2f}%", f"{duplicate_pct:.2f}%"],
        textposition="auto",
    ))
    fig.update_layout(
        title="Data Contamination Overview",
        yaxis_title="% of rows",
        template="plotly_white",
        height=350,
        margin=dict(t=50, b=40, l=40, r=20),
    )
    return fig
