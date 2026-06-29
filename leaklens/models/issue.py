from dataclasses import dataclass, field
from typing import Optional, Any, Dict

from .severity import Severity


@dataclass
class Issue:
    """A single finding from an analyzer. Analyzers only ever produce these —
    no printing, no scoring, no opinions about output format."""

    title: str
    severity: Severity
    message: str
    column: Optional[str] = None
    analyzer: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "severity": self.severity.value,
            "message": self.message,
            "column": self.column,
            "analyzer": self.analyzer,
            "details": self.details,
        }
