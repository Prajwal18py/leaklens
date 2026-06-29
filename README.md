# LeakLens

**Leakage & drift QA for ML datasets.** Think Ruff or ESLint, but for the things that quietly invalidate an ML experiment instead of the things that break your code.

Most EDA tools answer *"what does my data look like?"* LeakLens answers a different question: **"is my train/test split even valid?"**

```bash
pip install leaklens
```

## Why

The most common way an ML project fails isn't a bad model — it's a corrupted experiment. Target leakage, contaminated splits, drifted test sets, and preprocessing fit before the split all produce results that look great in your notebook and fall apart in production. LeakLens checks for exactly these failure modes, deterministically, with no invented "AI confidence scores."

## Quick start

```python
from leaklens import LeakLens

report = LeakLens(train_df, test_df, target="price").run()

report.summary()          # prints + returns a plain-text summary
report.to_html("report.html")   # full report with plotly visualizations
report.to_json()          # structured output for CI pipelines
report.issues             # raw list[Issue] if you want to handle them yourself
```

Works with a single dataframe too (drops the train/test-comparison checks automatically):

```python
report = LeakLens(df, target="churned").run()
```

Polars is supported — pass a `polars.DataFrame` and it's converted internally.

## What it checks

| Check | What it catches |
|---|---|
| **Target leakage** | A feature that maps almost 1:1 to the target, or has suspiciously high correlation / Cramér's V with it |
| **Train/test contamination** | Identical rows appearing in both train and test (the "sampled both sets from the same pool" bug) |
| **Distribution drift** | KS test (numeric columns) and PSI (categorical columns) between train and test |
| **Temporal leakage** | Test-set dates that fall before the latest date in train — a split that isn't actually chronological |
| **Duplicate columns** | Near-perfectly correlated numeric columns, or categorical columns with identical values (`price` vs `selling_price`) |
| **Schema mismatch** | Columns present in one set but not the other |
| **Dtype mismatch** | The same column typed differently between train and test |
| **Unseen categories** | Categories in test that never appeared in train (breaks most encoders) |
| **Constant / near-constant features** | Columns with zero or near-zero variance |
| **Preprocessing-before-split leakage** | Static analysis of a training script — flags `scaler.fit_transform(X)` called before `train_test_split()` |

No fairness scores, no "deployment readiness %," no invented metrics — every issue maps to a specific, explainable calculation.

## Preprocessing-leakage check (script analysis)

This one doesn't need a dataframe — it parses your training script's AST and checks call order:

```python
report = LeakLens(train_df, target="price", script="train.py").run()
```

```
❌ Line 6: 'scaler.fit(...)' is called before train_test_split() on line 8 —
   this leaks test data into preprocessing statistics.
```

## Configuration

Every threshold is tunable:

```python
from leaklens import LeakLens, Config

config = Config(
    target_corr_threshold=0.90,
    ks_alpha=0.01,
    psi_critical=0.20,
)
report = LeakLens(train_df, test_df, target="price", config=config).run()
```

## CLI (optional)

```bash
pip install leaklens[cli]
leaklens check train.csv test.csv --target price --report report.html
```

## Output shape

Every finding is a plain dataclass — no magic strings:

```python
from leaklens import Issue, Severity

Issue(
    title="Target Leakage",
    severity=Severity.CRITICAL,
    column="transaction_id",
    message="98.2% of values map to a single target value...",
    analyzer="target_leakage",
    details={"mapped_fraction": 0.982},
)
```

## Project layout

```
leaklens/
├── core.py              # LeakLens orchestrator
├── config.py            # tunable thresholds
├── exceptions.py
├── report.py             # HTML rendering
├── models/               # Issue, Report, Severity dataclasses
├── analyzers/            # one module per check, all return list[Issue]
├── visualizations/       # plotly figure builders
└── templates/            # report.html (jinja2)
```

Analyzers never print, plot, or score — they only emit `Issue` objects. Rendering is handled separately, so adding a new output format (or a new check) doesn't touch existing code.

## Roadmap

- **v1.5**: CLI polish, Jupyter widget output, more visualizations
- **v2.0**: GitHub Actions integration (fail a PR if leakage is detected), MLflow logging

## Development

```bash
pip install -e ".[dev]"
pytest --cov=leaklens
```

## License

MIT
