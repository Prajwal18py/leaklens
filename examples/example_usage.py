"""Quick demo: builds a deliberately broken train/test split and runs
LeakLens on it, so you can see every check fire at least once.

Run: python examples/example_usage.py
"""
import numpy as np
import pandas as pd
from leaklens import LeakLens

rng = np.random.default_rng(7)
n_train, n_test = 500, 150

train = pd.DataFrame({
    "customer_id": range(n_train),
    "age": rng.integers(18, 70, n_train),
    "city": rng.choice(["Delhi", "Mumbai", "Bengaluru"], n_train),
    "income": rng.normal(50000, 15000, n_train),
    "income_copy": None,  # filled below to be a near-duplicate column
    "signup_date": pd.date_range("2022-01-01", periods=n_train, freq="D"),
    "country": "India",  # constant feature
    "churned": rng.integers(0, 2, n_train),
})
train["income_copy"] = train["income"] * 1.0001  # duplicate-column bug
train["leaky_flag"] = train["churned"].map({0: "no", 1: "yes"})  # target leakage bug

test = pd.DataFrame({
    "customer_id": range(n_train - 50, n_train - 50 + n_test),  # overlaps with train -> contamination
    "age": rng.integers(18, 70, n_test),
    "city": rng.choice(["Delhi", "Mumbai", "Chennai"], n_test),  # 'Chennai' unseen in train
    "income": rng.normal(80000, 15000, n_test),  # drifted distribution
    "income_copy": None,
    "signup_date": pd.date_range("2022-06-01", periods=n_test, freq="D"),  # overlaps train dates
    "country": "India",
    "churned": rng.integers(0, 2, n_test),
})
test["income_copy"] = test["income"] * 1.0001
test["leaky_flag"] = test["churned"].map({0: "no", 1: "yes"})

if __name__ == "__main__":
    report = LeakLens(train, test, target="churned").run()
    report.summary()
    report.to_html("example_report.html")
    print("\nFull report saved to example_report.html")
