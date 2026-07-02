# LeakLens

**The pre-flight checklist for machine learning datasets.**

Most EDA tools answer *"what does my data look like?"* LeakLens answers a different question: **"is my train/test split even valid?"**

Think Ruff or ESLint — but for the structural mistakes that quietly invalidate an ML experiment instead of the ones that break your code.

```bash
pip install leaklens
```

[![PyPI version](https://badge.fury.io/py/leaklens.svg)](https://pypi.org/project/leaklens/)
[![Python](https://img.shields.io/pypi/pyversions/leaklens)](https://pypi.org/project/leaklens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
![LeakLens Dashboard](https://github.com/user-attachments/assets/bae7ed41-aa5d-425c-8cd1-d7e56bd92729)

![LeakLens Banner](https://github.com/user-attachments/assets/a9e60c31-b2f8-446b-ac44-16fbc9052ccf)

![LeakLens Clean](https://github.com/user-attachments/assets/596b88b3-6e63-4623-a76b-f6a2b5d1d33d)
## Why

The most common way an ML project fails isn't a bad model — it's a corrupted experiment. Target leakage, contaminated splits, drifted test sets, and preprocessing fit before the split all produce results that look great in your notebook and fall apart in production.

LeakLens catches exactly these failure modes — deterministically, with no invented scores or black-box metrics. Every finding maps to a specific, reproducible calculation you can explain in an interview or a code review.

---

## Quick start

```python
from leaklens import LeakLens

report = LeakLens(train_df, test_df, target="price").run()

report.summary()             # plain-text console output
report.to_html("report.html")  # full interactive dashboard
report.to_json()             # structured output for CI pipelines
report.issues                # raw List[Issue] for custom handling
```

Works with a single dataframe too — train/test comparison checks are skipped automatically:

```python
report = LeakLens(df, target="churned").run()
```

Polars DataFrames are supported — passed in as-is and converted internally.

---

## What it checks

| Check | Method | What it catches |
|---|---|---|
| **Target Leakage** | Correlation / Cramer's V / group mapping | Features that directly or indirectly encode the target |
| **Missing Value Leakage** | Point-biserial r / Cramer's V on missingness indicator | Columns where *whether* a value is missing correlates with the target |
| **Train/Test Contamination** | Full-row hash matching | Identical rows appearing in both splits |
| **Distribution Drift** | KS test (numeric) + PSI (categorical) | Train and test drawn from different distributions |
| **Temporal Leakage** | Date-column auto-detection + chronology check | Test-set dates that fall before the latest train date |
| **Duplicate Columns** | Correlation threshold + Series.equals() | Near-identical columns adding redundant collinearity |
| **Schema Validation** | Column diff + dtype comparison + category sets | Missing columns, dtype mismatches, unseen categories, constant features |
| **Preprocessing Leakage** | AST static analysis | Scalers, encoders, PCA, SMOTE etc. fit before `train_test_split()` |

Every finding includes:
- The raw metric value that triggered it (KS statistic, PSI, correlation, etc.)
- The exact threshold it crossed (from your `Config`, not hidden)
- **Root cause analysis** for drift findings (mean shift, variance shift, 95th-percentile shift)
- A plain-English "why it matters" explanation
- Concrete fix steps

No fairness scores, no deployment readiness %, no AI confidence badges — just real statistics.

---

## The report

`report.to_html("report.html")` generates a self-contained interactive dashboard:

- **Verdict banner** — `DO NOT TRAIN` / `TRAIN WITH CAUTION` / `SAFE TO TRAIN`, derived directly from critical/warning counts
- **Dataset fingerprint** — schema hash + distribution hash in the header, so you can confirm two runs used identical data
- **Full drift ranking** — every checked column shown, stable ones in grey, drifted ones in red/amber — not just the ones that failed
- **KDE distribution overlays** — smooth kernel density curves for numeric drift, not blurry histograms
- **Root cause bullets** — "Mean increased (29.4 → 55.3), 95th percentile shifted (54.9 → 83.5)"
- **Expandable issue accordions** — detected value, threshold used, why it matters, fix steps
- **Severity-colored recommendation cards** — one per issue type, concise and scannable
- **JSON export** — one-click download of all findings as structured JSON
- **Print / Save as PDF** — properly handles collapsed accordions, chart resolution, and color preservation

---

## Root cause analysis

For numeric drift findings, LeakLens explains *what shape* of drift occurred — not just that it happened:

```
❌ Distribution Drift [Age]
   KS p-value = 0.0000

   Likely Cause:
   ✔ Mean increased (29.4 → 55.3)
   ✔ 95th percentile shifted (54.9 → 83.5)
```

Pure statistics, no model involved.

---

## Missing value leakage

A check most libraries don't have — detects when the *absence* of a value is itself predictive:

```
⚠ Missing Value Leakage [Cabin]
  Whether 'Cabin' is missing correlates with the target
  (point-biserial r = -0.313, p < 0.0001)
```

In the Titanic dataset, this fires automatically — cabin missingness genuinely correlates with survival, and any imputation strategy that fills in a "typical" value would silently erase that signal or inadvertently leak it.

---

## Dataset fingerprint

Every report includes a schema hash and distribution hash:

```python
report.meta["train_fingerprint"]
# {'schema_hash': 'e0cf60009fb642c4', 'distribution_hash': '78e61c68b140fe5b',
#  'n_rows': 624, 'n_columns': 14}
```

Useful for confirming that two training runs actually used the same data, not silently different versions of a file.

---

## Preprocessing leakage (AST check)

Parses your training script's syntax tree — no execution needed:

```python
report = LeakLens(train_df, target="price", script="train.py").run()
```

Catches any of these fit before `train_test_split()`:

```
StandardScaler  MinMaxScaler  RobustScaler
PCA  TruncatedSVD  KernelPCA
KNNImputer  SimpleImputer  IterativeImputer
LabelEncoder  OneHotEncoder  OrdinalEncoder  TargetEncoder
SMOTE  ADASYN  RandomOverSampler  RandomUnderSampler
SelectKBest  SelectPercentile  RFE  VarianceThreshold
```

---

## Configuration

Every threshold is tunable — nothing is hardcoded:

```python
from leaklens import LeakLens, Config

config = Config(
    target_corr_threshold=0.90,      # default 0.95
    ks_alpha=0.01,                   # default 0.05
    psi_critical=0.20,               # default 0.25
    missingness_corr_threshold=0.25, # default 0.30
)
report = LeakLens(train_df, test_df, target="price", config=config).run()
```

---

## CLI + GitHub Actions

```bash
pip install "leaklens[cli]"
leaklens train.csv test.csv --target price --report report.html --fail-on critical
```

`--fail-on` controls CI exit code: `critical` (default), `warning`, or `none`.

Drop this into `.github/workflows/leaklens-qa.yml` to fail PRs that introduce data leakage:

```yaml
- name: LeakLens QA
  run: |
    pip install "leaklens[cli]"
    leaklens data/train.csv data/test.csv \
      --target price \
      --report leaklens_report.html \
      --fail-on critical

- name: Upload report
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: leaklens-report
    path: leaklens_report.html
```

A full workflow template is included at `.github/workflows/leaklens-qa.yml`.

---

## Output shape

Every finding is a structured dataclass — no magic strings, no parsing required:

```python
Issue(
    title="Distribution Drift",
    severity=Severity.CRITICAL,
    column="Age",
    analyzer="drift",
    message="KS test p-value=0.0000 — distributions differ significantly.",
    details={
        "ks_stat": 0.660,
        "p_value": 0.0,
        "root_cause": ["Mean increased (29.4 → 55.3)", "95th percentile shifted (54.9 → 83.5)"],
    },
)
```

---

## Project layout

```
leaklens/
├── core.py                   # LeakLens orchestrator
├── config.py                 # all thresholds, fully overridable
├── exceptions.py
├── report.py                 # HTML rendering via Jinja2
├── cli.py                    # optional Typer CLI
├── utils.py                  # to_pandas(), fingerprint, identifier detection
├── models/                   # Issue, Report, Severity dataclasses
├── analyzers/                # one module per check, all return List[Issue]
│   ├── target_leakage.py
│   ├── contamination.py
│   ├── drift.py              # KS + PSI + root cause analysis
│   ├── temporal.py
│   ├── duplicate_columns.py
│   ├── schema.py
│   ├── missing_value_leakage.py
│   └── preprocessing.py      # AST-based static analysis
├── visualizations/           # Plotly KDE figure builders
└── templates/
    └── report.html           # full dashboard, single Jinja2 template
```

Analyzers never print, plot, or score — they only emit `Issue` objects. Rendering is handled separately, so adding a new output format or a new check doesn't touch existing detection logic.

---

## Installation

```bash
# Core library
pip install leaklens

# With CLI (for GitHub Actions / terminal use)
pip install "leaklens[cli]"

# For development
pip install "leaklens[dev]"
```

**Requirements:** Python 3.9+, pandas, numpy, scipy, plotly, jinja2

---

## Development

```bash
git clone https://github.com/Prajwal18py/leaklens
cd leaklens
pip install -e ".[dev]"
pytest tests/ -v
```

37 tests, one synthetic case per analyzer behavior.

---

## What LeakLens is NOT

- Not an EDA library (use ydata-profiling, Sweetviz for that)
- Not an AutoML tool
- Not a model trainer or evaluator
- Not a fairness auditor
- No LLM calls, no black-box scores, no invented metrics

LeakLens does one thing: validates that your train/test split is structurally sound before you train anything on it.

---

## License

MIT © Prajwal A.

[GitHub](https://github.com/Prajwal18py/leaklens) · [PyPI](https://pypi.org/project/leaklens/) · [Issues](https://github.com/Prajwal18py/leaklens/issues)