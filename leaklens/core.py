import time
from typing import Optional, List, Dict, Any

from scipy.stats import ks_2samp

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
from .visualizations.drift import numeric_drift_kde_figure

ANALYZER_DISPLAY_NAMES = {
    "target_leakage": "Target Leakage",
    "contamination": "Train/Test Contamination",
    "drift": "Distribution Drift",
    "temporal_leakage": "Temporal Leakage",
    "duplicate_columns": "Duplicate Columns",
    "schema": "Schema Validation",
    "preprocessing_leakage": "Preprocessing Leakage",
}

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

# Short, generic "why this matters" context shown inside each issue's
# expanded panel — not detection logic, purely explanatory copy.
WHY_IT_MATTERS = {
    "Target Leakage": "If a feature leaks the target, your model will look accurate during evaluation but fail in production, where that feature won't carry the same information.",
    "Distribution Drift": "Models trained on one distribution often perform worse when evaluated on data with a different shape — your test metrics may not reflect real-world performance.",
    "Train/Test Contamination": "If the same rows appear in both sets, your model has effectively already seen part of the test set — reported accuracy will be inflated.",
    "Duplicate Rows": "Duplicate rows can bias a model toward over-represented examples and inflate validation scores.",
    "Temporal Leakage": "If test data isn't strictly after train data in time, the model may be evaluated on information it wouldn't have had at training time.",
    "Duplicate Columns": "Near-duplicate features add collinearity without new information, which can destabilize some models and waste training time.",
    "Schema Mismatch": "A model trained on one column set will fail outright when scored on a dataset with a different schema.",
    "Dtype Mismatch": "Type mismatches between train and test commonly cause silent encoding bugs or outright crashes at inference time.",
    "Unseen Categories": "Most encoders (one-hot, label encoding) fail or silently produce wrong results on categories they never saw during fit.",
    "Constant Feature": "A column with no variance contributes zero predictive signal and only adds noise/dimensionality.",
    "Near-Constant Feature": "Very low-variance columns rarely help a model and can be safely dropped after a quick check.",
}


# Short checklist-style fix steps shown inside each recommendation card —
# presentation copy only, derived from the issue, not part of detection logic.
FIX_STEPS = {
    "Target Leakage": ["Verify the column isn't derived from the target", "Drop or re-engineer the column", "Re-run checks after removing it"],
    "Distribution Drift": ["Verify the train/test split was done correctly", "Check if the column is an unintended ID/index", "Re-sample or retrain with newer data"],
    "Train/Test Contamination": ["Re-run your split with deduplication", "Check for shared sampling pools between sets", "Remove overlapping rows before evaluating"],
    "Duplicate Rows": ["Call df.drop_duplicates() before training", "Investigate why duplicates exist upstream"],
    "Temporal Leakage": ["Sort by timestamp before splitting", "Use a strict cutoff date for train vs test", "Avoid random splits on time-ordered data"],
    "Duplicate Columns": ["Inspect both columns to confirm redundancy", "Drop the less interpretable of the two"],
    "Schema Mismatch": ["Diff train/test columns directly", "Align column names and ordering before training"],
    "Dtype Mismatch": ["Cast both columns to the same dtype", "Check the data loading step for inconsistencies"],
    "Unseen Categories": ["Use an encoder that handles unknowns (e.g. handle_unknown='ignore')", "Add an explicit 'unknown' bucket", "Re-check category coverage between sets"],
    "Constant Feature": ["Drop the column — it adds no signal", "Confirm it isn't a data loading bug (e.g. wrong column selected)"],
    "Near-Constant Feature": ["Review the column's variance", "Consider dropping if not domain-critical"],
}


ISSUE_ICONS = {
    "Target Leakage": "&#127919;",
    "Distribution Drift": "&#128200;",
    "Train/Test Contamination": "&#129516;",
    "Duplicate Rows": "&#128203;",
    "Temporal Leakage": "&#128337;",
    "Duplicate Columns": "&#128230;",
    "Schema Mismatch": "&#129513;",
    "Schema Validation": "&#129513;",
    "Dtype Mismatch": "&#129513;",
    "Unseen Categories": "&#128194;",
    "Constant Feature": "&#9888;",
    "Near-Constant Feature": "&#9888;",
    "Preprocessing Leakage": "&#128295;",
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
        self.script = script
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
        report.meta["drift_ranking"] = self._build_full_drift_ranking(report)
        report.meta["high_cardinality_cards"] = self._build_high_cardinality_cards()
        report.meta["data_quality_strip"] = self._build_data_quality_strip(report)
        report.meta["recommendations"] = self._build_recommendations(report)
        report.meta["overlap_stats"] = self._build_overlap_stats(report)
        report.meta["numeric_drift_figures"] = self._build_numeric_drift_figures(report)
        report.meta["why_it_matters"] = WHY_IT_MATTERS
        report.meta["fix_steps"] = FIX_STEPS
        report.meta["issue_icons"] = ISSUE_ICONS
        report.meta["risk_label"] = self._compute_risk_label(report)
        report.meta["verdict"], report.meta["verdict_reason"], report.meta["primary_risks"] = self._compute_verdict(report)
        report.meta["thresholds"] = self._build_thresholds()
        report.meta["export_json"] = self._build_export_json(report)

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

    def _build_full_drift_ranking(self, report: Report, top_n: int = 10) -> List[Dict[str, Any]]:
        """Unlike v1, this ranks EVERY checked column (not just flagged
        ones), so a single drifted column doesn't leave the panel looking
        empty — stable columns show up too, just visually de-emphasized."""
        if self.test is None:
            return []

        numeric_cols = set(safe_numeric_columns(self.train))
        flagged = {i.column: i for i in report.issues if i.title == "Distribution Drift"}
        common_cols = [c for c in self.train.columns if c in self.test.columns and c != self.target]

        rows = []
        for col in common_cols:
            if col in numeric_cols and col in safe_numeric_columns(self.test):
                train_clean = self.train[col].dropna()
                test_clean = self.test[col].dropna()
                if len(train_clean) < 5 or len(test_clean) < 5:
                    continue
                stat, _ = ks_2samp(train_clean, test_clean)
                metric_label, value = "KS", float(stat)
            else:
                if is_likely_identifier_column(
                    self.train[col],
                    self.config.high_cardinality_ratio_threshold,
                    self.config.high_cardinality_absolute_threshold,
                ):
                    continue
                psi = DriftAnalyzer._calculate_psi(self.train[col], self.test[col])
                if psi is None:
                    continue
                metric_label, value = "PSI", float(psi)

            issue = flagged.get(col)
            severity = issue.severity.value if issue else "stable"
            rows.append({"column": col, "metric_label": metric_label, "value": value, "severity": severity})

        rows.sort(key=lambda r: (r["severity"] != "critical", r["severity"] != "warning", -r["value"]))
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

        constant_count = sum(1 for i in report.issues if i.title in ("Constant Feature", "Near-Constant Feature"))
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
            fig = numeric_drift_kde_figure(col, self.train[col], self.test[col])
            figures.append({
                "column": col,
                "html": fig.to_html(full_html=False, include_plotlyjs=False),
                "ks_stat": round(issue.details.get("ks_stat", 0), 4),
                "p_value": round(issue.details.get("p_value", 0), 4),
                "status": "Drift Detected" if issue.severity in (Severity.CRITICAL, Severity.WARNING) else "Stable",
            })
        return figures

    @staticmethod
    def _compute_risk_label(report: Report) -> str:
        if len(report.critical) > 0:
            return "HIGH RISK"
        if len(report.warnings) > 0:
            return "MEDIUM RISK"
        return "LOW RISK"

    @staticmethod
    def _compute_verdict(report: Report):
        """A one-line plain-English verdict, directly derived from the
        critical/warning counts already on the report — not a new metric,
        just a clearer restatement of data that's already there."""
        critical_issues = report.critical
        if critical_issues:
            primary_risks = list(dict.fromkeys(i.title for i in critical_issues))  # dedup, preserve order
            reason = f"{len(critical_issues)} critical issue(s) found."
            return "DO NOT TRAIN", reason, primary_risks
        if report.warnings:
            primary_risks = list(dict.fromkeys(i.title for i in report.warnings))
            return "TRAIN WITH CAUTION", f"{len(report.warnings)} warning(s) found — review before relying on results.", primary_risks
        return "SAFE TO TRAIN", "All checks passed — no leakage, drift, or schema issues detected.", []

    def _build_thresholds(self) -> Dict[str, float]:
        """Snapshot of the actual Config values used for this run, so the
        report can show 'Detected vs Threshold' honestly — every number here
        is a real value that drove a real decision, not decoration."""
        c = self.config
        return {
            "target_corr_threshold": c.target_corr_threshold,
            "cramers_v_threshold": c.cramers_v_threshold,
            "near_identical_mapping_threshold": 0.98,  # hardcoded in TargetLeakageAnalyzer
            "psi_warning": c.psi_warning,
            "psi_critical": c.psi_critical,
            "ks_alpha": c.ks_alpha,
            "contamination_warning_pct": c.contamination_warning_pct,
            "contamination_critical_pct": c.contamination_critical_pct,
            "duplicate_column_corr_threshold": c.duplicate_column_corr_threshold,
            "constant_feature_threshold": c.constant_feature_threshold,
        }

    def _build_export_json(self, report: Report) -> str:
        """A lean JSON export (issues + summary only) — deliberately
        excludes the embedded Plotly figure HTML from the full report.meta,
        since that would bloat the exported payload with megabytes of
        duplicated chart markup."""
        import json
        data = {
            "dataset_name": self.dataset_name,
            "target": self.target,
            "n_train_rows": len(self.train),
            "n_test_rows": len(self.test) if self.test is not None else None,
            "verdict": report.meta.get("verdict"),
            "verdict_reason": report.meta.get("verdict_reason"),
            "critical_count": len(report.critical),
            "warning_count": len(report.warnings),
            "checks_run": report.checks_run,
            "issues": [i.to_dict() for i in report.issues],
        }
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.replace("</", "<\\/")