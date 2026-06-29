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

    return template.render(
        issues=sorted_issues,
        critical_count=len(report.critical),
        warning_count=len(report.warnings),
        info_count=len(report.infos),
        checks_run=report.checks_run,
        figures_html=report.meta.get("figures_html", []),
        meta=report.meta,
    )
