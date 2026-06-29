from typing import Optional, List

from .config import Config
from .exceptions import InvalidTargetError, EmptyDataFrameError
from .utils import to_pandas, safe_numeric_columns
from .models.report import Report
from .analyzers.target_leakage import TargetLeakageAnalyzer
from .analyzers.contamination import ContaminationAnalyzer
from .analyzers.drift import DriftAnalyzer
from .analyzers.temporal import TemporalLeakageAnalyzer
from .analyzers.duplicate_columns import DuplicateColumnsAnalyzer
from .analyzers.schema import SchemaAnalyzer
from .analyzers.preprocessing import PreprocessingLeakageAnalyzer
from .visualizations.drift import numeric_drift_figure, categorical_drift_figure
from .visualizations.contamination import contamination_summary_figure


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
    ):
        self.train = to_pandas(train)
        self.test = to_pandas(test) if test is not None else None
        self.target = target
        self.config = config or Config()
        self.script = script  # optional source/path for the preprocessing-leak check

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
        report = Report()
        for analyzer in self._analyzers:
            report.issues.extend(analyzer.run(self.train, self.test, self.target))
            report.checks_run.append(analyzer.name)

        if self.script:
            pp_analyzer = PreprocessingLeakageAnalyzer(self.config)
            report.issues.extend(pp_analyzer.analyze_script(self.script))
            report.checks_run.append(pp_analyzer.name)

        report.meta["n_train_rows"] = len(self.train)
        report.meta["n_test_rows"] = len(self.test) if self.test is not None else None
        report.meta["target"] = self.target
        report.meta["figures_html"] = self._build_figures_html(report)

        return report

    def _build_figures_html(self, report: Report) -> List[str]:
        figures = []
        if self.test is None:
            return figures

        numeric_cols = set(safe_numeric_columns(self.train))
        drift_cols = [i.column for i in report.issues if i.title == "Distribution Drift" and i.column]
        seen = set()
        for col in drift_cols:
            if col in seen or col not in self.train.columns or col not in self.test.columns:
                continue
            seen.add(col)
            if len(seen) > 6:  # keep the report light
                break
            if col in numeric_cols:
                fig = numeric_drift_figure(col, self.train[col], self.test[col])
            else:
                fig = categorical_drift_figure(col, self.train[col], self.test[col])
            figures.append(fig.to_html(full_html=False, include_plotlyjs=False))

        contamination_issue = next((i for i in report.issues if i.title == "Train/Test Contamination"), None)
        dup_issue = next((i for i in report.issues if i.title == "Duplicate Rows"), None)
        if contamination_issue or dup_issue:
            overlap_pct = contamination_issue.details.get("overlap_pct", 0) if contamination_issue else 0
            dup_pct = dup_issue.details.get("duplicate_pct", 0) if dup_issue else 0
            fig = contamination_summary_figure(overlap_pct, dup_pct)
            figures.append(fig.to_html(full_html=False, include_plotlyjs=False))

        return figures
