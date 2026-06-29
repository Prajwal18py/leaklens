import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

from .issue import Issue
from .severity import Severity


@dataclass
class Report:
    """Holds every Issue found during a run, plus metadata. Rendering (HTML /
    JSON / console) is delegated out so analyzers and this model stay
    decoupled from presentation."""

    issues: List[Issue] = field(default_factory=list)
    checks_run: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    def by_severity(self, severity: Severity) -> List[Issue]:
        return [i for i in self.issues if i.severity == severity]

    @property
    def critical(self) -> List[Issue]:
        return self.by_severity(Severity.CRITICAL)

    @property
    def warnings(self) -> List[Issue]:
        return self.by_severity(Severity.WARNING)

    @property
    def infos(self) -> List[Issue]:
        return self.by_severity(Severity.INFO)

    def summary(self) -> str:
        lines = [
            "LeakLens Report",
            "=" * 40,
            f"Checks run: {len(self.checks_run)}",
            f"Critical issues: {len(self.critical)}",
            f"Warnings: {len(self.warnings)}",
            f"Info: {len(self.infos)}",
            "",
        ]
        order = {"critical": 0, "warning": 1, "info": 2}
        for issue in sorted(self.issues, key=lambda i: order.get(i.severity.value, 3)):
            marker = {"critical": "\u274c", "warning": "\u26a0", "info": "\u2139"}.get(issue.severity.value, "-")
            col = f" [{issue.column}]" if issue.column else ""
            lines.append(f"{marker} {issue.title}{col}: {issue.message}")
        if not self.issues:
            lines.append("\u2714 No issues detected.")
        text = "\n".join(lines)
        print(text)
        return text

    def to_json(self, path: str = None) -> str:
        data = {
            "meta": self.meta,
            "checks_run": self.checks_run,
            "summary": {
                "critical": len(self.critical),
                "warning": len(self.warnings),
                "info": len(self.infos),
            },
            "issues": [i.to_dict() for i in self.issues],
        }
        json_str = json.dumps(data, indent=2, default=str)
        if path:
            Path(path).write_text(json_str, encoding="utf-8")
        return json_str

    def to_html(self, path: str = "report.html") -> str:
        # Deferred import avoids a circular import at module load time
        # (leaklens.report renders this exact Report object).
        from ..report import render_html

        html = render_html(self)
        Path(path).write_text(html, encoding="utf-8")
        return path
