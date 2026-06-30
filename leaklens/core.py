import time
from typing import Optional, List, Dict, Any

from .config import Config
from .exceptions import InvalidTargetError, EmptyDataFrameError
from .utils import to_pandas, safe_numeric_columns, is_likely_identifier_column
from .models.report import Report
from .models.severity import Severity
from .analyzers.target_leakage import TargetLeakageAnalyzer
from .analyzers.contamination import ContaminationAnalyzer
from .analyzers.drift import DriftAnalyzer
from .analyzers.temporal import TemporalLeakageAnalyzer
from .analyzers.duplicate_columns import DuplicateColumnsAnalyzer
from .analyzers.schema import SchemaAnalyzer
from .analyzers.preprocessing import PreprocessingLeakageAnalyzer
from .visualizations.drift import numeric_drift_figure

ANALYZER_DISPLAY_NAMES = {
    "target_leakage": "Target Leakage",
    "contamination": "Train/Test Contamination",
    "drift": "Distribution Drift",
    "temporal_leakage": "Temporal Leakage",
    "duplicate_columns": "Duplicate Columns",
    "schema": "Schema Validation",
    "preprocessing_leakage": "Preprocessing Leakage",
}

# Maps an issue title -> a generic, actionable recommendation. Kept separate
# from the analyzers themselves (analyzers only ever emit facts); this is
# presentation-layer advice, not part of the detection logic.
RECOMMENDATION_TEMPLATES = {
    "Target Leakage": "Investigate and likely drop '{col}' before training — it appears to leak the target.",
    "Distribution Drift": "Distribution drift detected in '{col}'. Re-sample test data or retrain on more recent data.",
    "Train/Test Contamination": "Remove duplicate rows shared between train and test — current metrics may be inflated.",
    "Duplicate Rows": "Deduplicate the training set before fitting a model.",
    "Temporal Leakage": "Switch to a chronological split so test data never precedes train data.",
    "Duplicate Columns": "Drop one of the duplicate columns to avoid redundant, collinear features.",
    "Schema Mismatch": "Align column names between train and test before training.",
    "Dtype Mismatch": "Cast '{col}' to a matching dtype in both train and test.",
    "Unseen Categories": "Handle unseen categories in '{col}' with an 'unknown' bucket or a robust encoder.",
    "Constant Feature": "Drop '{col}' — it carries no information.",
    "Near-Constant Feature": "Review '{col}' — it has very low variance and may not be useful.",
}


class LeakLens:
    """Inspects a train/test split (or a single dataset) for the issues that
    most commonly invalidate an ML experiment: target leakage, train/test
    contamination, distribution drift, temporal leakage, schema mismatches,
    and accidental near-duplicate columns.

    Example
    -------
    >>> from leaklens import LeakLens
    >>> report = LeakLens(train_df, test_df, target="price").run()
    >>> report.summary()
    >>> report.to_html("report.html")
    """

    def __init__(
        self,
        train,
        test=None,
        target: Optional[str] = None,
        config: Optional[Config] = None,
        script: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ):
        self.train = to_pandas(train)
        self.test = to_pandas(test) if test is not None else None
        self.target = target
        self.config = config or Config()
        self.script = script  # optional source/path for the preprocessing-leak check
        self.dataset_name = dataset_name or "dataset"

        if self.train is None or len(self.train) == 0:
            raise EmptyDataFrameError("train dataframe is empty.")
        if self.target is not None and self.target not in self.train.columns:
            raise InvalidTargetError(f"target column '{self.target}' not found in train dataframe.")

        self._analyzers = [
            TargetLeakageAnalyzer(self.config),
            ContaminationAnalyzer(self.config),
            DriftAnalyzer(self.config),
            TemporalLeakageAnalyzer(self.config),
            DuplicateColumnsAnalyzer(self.config),
            SchemaAnalyzer(self.config),
        ]

    def run(self) -> Report:
        start = time.perf_counter()
        report = Report()
        analyzer_for_check: Dict[str, List] = {}

        for analyzer in self._analyzers:
            issues = analyzer.run(self.train, self.test, self.target)
            report.issues.extend(issues)
            report.checks_run.append(analyzer.name)
            analyzer_for_check[analyzer.name] = issues

        if self.script:
            pp_analyzer = PreprocessingLeakageAnalyzer(self.config)
            pp_issues = pp_analyzer.analyze_script(self.script)
            report.issues.extend(pp_issues)
            report.checks_run.append(pp_analyzer.name)
            analyzer_for_check[pp_analyzer.name] = pp_issues

        runtime = time.perf_counter() - start

        report.meta["dataset_name"] = self.dataset_name
        report.meta["n_train_rows"] = len(self.train)
        report.meta["n_test_rows"] = len(self.test) if self.test is not None else None
        report.meta["target"] = self.target
        report.meta["runtime_seconds"] = round(runtime, 3)
        report.meta["version"] = self._get_version()

        report.meta["checks_status"] = self._build_checks_status(analyzer_for_check)
        report.meta["drift_ranking"] = self._build_drift_ranking(report)
        report.meta["high_cardinality_cards"] = self._build_high_cardinality_cards()
        report.meta["data_quality_strip"] = self._build_data_quality_strip(report)
        report.meta["recommendations"] = self._build_recommendations(report)
        report.meta["overlap_stats"] = self._build_overlap_stats(report)
        report.meta["numeric_drift_figures"] = self._build_numeric_drift_figures(report)
        score, label = self._compute_risk_score(report)
        report.meta["risk_score"] = score
        report.meta["risk_label"] = label

        return report

    @staticmethod
    def _get_version() -> str:
        try:
            from . import __version__
            return __version__
        except ImportError:
            return "0.0.0"

    # ── Aggregation helpers ────────────────────────────────────────────────

    def _build_checks_status(self, analyzer_for_check: Dict[str, List]) -> List[Dict[str, Any]]:
        rows = []
        for name, issues in analyzer_for_check.items():
            if any(i.severity == Severity.CRITICAL for i in issues):
                status = "critical"
            elif any(i.severity == Severity.WARNING for i in issues):
                status = "warning"
            else:
                status = "passed"
            rows.append({
                "name": ANALYZER_DISPLAY_NAMES.get(name, name),
                "status": status,
                "count": len(issues),
            })
        return rows

    def _build_drift_ranking(self, report: Report, top_n: int = 8) -> List[Dict[str, Any]]:
        drift_issues = [i for i in report.issues if i.title == "Distribution Drift"]
        rows = []
        for i in drift_issues:
            if "psi" in i.details:
                metric_label, value = "PSI", i.details["psi"]
            elif "ks_stat" in i.details:
                metric_label, value = "KS", i.details["ks_stat"]
            else:
                continue
            rows.append({
                "column": i.column,
                "metric_label": metric_label,
                "value": value,
                "severity": i.severity.value,
            })
        rows.sort(key=lambda r: r["value"], reverse=True)
        rows = rows[:top_n]
        max_val = max((r["value"] for r in rows), default=1) or 1
        for r in rows:
            r["bar_pct"] = round(min(r["value"] / max_val, 1.0) * 100, 1)
        return rows

    def _build_high_cardinality_cards(self) -> List[Dict[str, Any]]:
        cards = []
        cat_cols = self.train.select_dtypes(exclude="number").columns
        for col in cat_cols:
            if col == self.target:
                continue
            if not is_likely_identifier_column(
                self.train[col],
                self.config.high_cardinality_ratio_threshold,
                self.config.high_cardinality_absolute_threshold,
            ):
                continue
            train_unique = set(self.train[col].dropna().unique())
            unseen_count = 0
            if self.test is not None and col in self.test.columns:
                test_unique = set(self.test[col].dropna().unique())
                unseen_count = len(test_unique - train_unique)
            cards.append({
                "column": col,
                "unique_values": len(train_unique),
                "unseen_count": unseen_count,
                "recommendation": "Drop, hash, or encode carefully — too high-cardinality for one-hot encoding.",
            })
        return cards

    def _build_data_quality_strip(self, report: Report) -> Dict[str, Any]:
        missing_train_pct = float(self.train.isna().mean().mean() * 100) if len(self.train.columns) else 0.0
        missing_test_pct = None
        if self.test is not None and len(self.test.columns):
            missing_test_pct = float(self.test.isna().mean().mean() * 100)

        dup_issue = next((i for i in report.issues if i.title == "Duplicate Rows"), None)
        contamination_issue = next((i for i in report.issues if i.title == "Train/Test Contamination"), None)
        dup_count = (dup_issue.details.get("duplicate_count", 0) if dup_issue else 0) + \
                    (contamination_issue.details.get("overlap_count", 0) if contamination_issue else 0)
        total_rows = len(self.train) + (len(self.test) if self.test is not None else 0)
        dup_pct = (dup_count / total_rows * 100) if total_rows else 0.0

        constant_count = sum(
            1 for i in report.issues if i.title in ("Constant Feature", "Near-Constant Feature")
        )
        dtype_mismatch_count = sum(1 for i in report.issues if i.title == "Dtype Mismatch")
        high_card_count = len(self._build_high_cardinality_cards())
        total_cols = max(len(self.train.columns), 1)

        return {
            "missing_train_pct": round(missing_train_pct, 1),
            "missing_test_pct": round(missing_test_pct, 1) if missing_test_pct is not None else None,
            "duplicate_count": int(dup_count),
            "duplicate_pct": round(dup_pct, 1),
            "constant_count": constant_count,
            "constant_pct": round(constant_count / total_cols * 100, 1),
            "high_cardinality_count": high_card_count,
            "high_cardinality_pct": round(high_card_count / total_cols * 100, 1),
            "dtype_mismatch_count": dtype_mismatch_count,
            "dtype_mismatch_pct": round(dtype_mismatch_count / total_cols * 100, 1),
        }

    def _build_recommendations(self, report: Report, limit: int = 6) -> List[str]:
        order = {"critical": 0, "warning": 1, "info": 2}
        sorted_issues = sorted(report.issues, key=lambda i: order.get(i.severity.value, 3))
        seen_titles = set()
        recs = []
        for issue in sorted_issues:
            key = (issue.title, issue.column)
            if key in seen_titles:
                continue
            seen_titles.add(key)
            template = RECOMMENDATION_TEMPLATES.get(issue.title)
            if not template:
                continue
            recs.append(template.format(col=issue.column or ""))
            if len(recs) >= limit:
                break
        return recs

    def _build_overlap_stats(self, report: Report) -> Dict[str, Any]:
        contamination_issue = next((i for i in report.issues if i.title == "Train/Test Contamination"), None)
        overlap_count = contamination_issue.details.get("overlap_count", 0) if contamination_issue else 0
        overlap_pct = contamination_issue.details.get("overlap_pct", 0.0) if contamination_issue else 0.0
        return {
            "train_rows": len(self.train),
            "test_rows": len(self.test) if self.test is not None else 0,
            "overlap_count": int(overlap_count),
            "overlap_pct": round(overlap_pct, 1),
        }

    def _build_numeric_drift_figures(self, report: Report, top_n: int = 4) -> List[Dict[str, Any]]:
        if self.test is None:
            return []
        numeric_cols = set(safe_numeric_columns(self.train))
        drift_issues = [
            i for i in report.issues
            if i.title == "Distribution Drift" and i.column in numeric_cols and "ks_stat" in i.details
        ]
        drift_issues.sort(key=lambda i: i.details.get("ks_stat", 0), reverse=True)
        figures = []
        for issue in drift_issues[:top_n]:
            col = issue.column
            if col not in self.test.columns:
                continue
            fig = numeric_drift_figure(col, self.train[col], self.test[col])
            figures.append({
                "column": col,
                "html": fig.to_html(full_html=False, include_plotlyjs=False),
                "ks_stat": round(issue.details.get("ks_stat", 0), 4),
                "p_value": round(issue.details.get("p_value", 0), 4),
                "status": "Drift Detected" if issue.severity in (Severity.CRITICAL, Severity.WARNING) else "Stable",
            })
        return figures

    @staticmethod
    def _compute_risk_score(report: Report):
        """Transparent, fully-documented score — NOT a black-box metric.
        Formula: 100 - 20*critical_issues - 5*warning_issues, floored at 0.
        This is shown in the report itself so it's never an unexplained number.
        """
        critical = len(report.critical)
        warning = len(report.warnings)
        score = max(0, min(100, 100 - critical * 20 - warning * 5))
        if critical > 0:
            label = "HIGH RISK"
        elif warning > 0:
            label = "MEDIUM RISK"
        else:
            label = "LOW RISK"
        return score, label