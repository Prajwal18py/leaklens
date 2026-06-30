"""Top-level HTML rendering. Kept separate from leaklens.models.report.Report
so the data model never has to know about jinja2 or plotly directly — it
just calls into here via a deferred import."""

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path(__file__).parent / "templates"


def render_html(report) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html")

    order = {"critical": 0, "warning": 1, "info": 2}
    sorted_issues = sorted(report.issues, key=lambda i: order.get(i.severity.value, 3))
    checks_status = report.meta.get("checks_status", [])
    passed_count = sum(1 for c in checks_status if c["status"] == "passed")

    return template.render(
        issues=sorted_issues,
        critical_count=len(report.critical),
        warning_count=len(report.warnings),
        passed_count=passed_count,
        checks_run=report.checks_run,
        checks_status=checks_status,
        drift_ranking=report.meta.get("drift_ranking", []),
        high_cardinality_cards=report.meta.get("high_cardinality_cards", []),
        data_quality=report.meta.get("data_quality_strip", {}),
        recommendations=report.meta.get("recommendations", []),
        overlap=report.meta.get("overlap_stats", {}),
        numeric_drift_figures=report.meta.get("numeric_drift_figures", []),
        why_it_matters=report.meta.get("why_it_matters", {}),
        fix_steps=report.meta.get("fix_steps", {}),
        risk_label=report.meta.get("risk_label", "LOW RISK"),
        verdict=report.meta.get("verdict", "SAFE TO TRAIN"),
        verdict_reason=report.meta.get("verdict_reason", ""),
        meta=report.meta,
    )