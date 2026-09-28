"""
Task 1 — Kaggle Titanic survival classification: A/B test of my solution vs. a
published baseline.

Dataset : Kaggle "Titanic - Machine Learning from Disaster" (train.csv, 891
          passengers). Downloaded here from a well-known public mirror of the
          exact same file (https://raw.githubusercontent.com/datasciencedojo/
          datasets/master/titanic.csv) since this session has no Kaggle
          account/API token to authenticate a direct competition download.
          Row count (891) and columns match the official competition file.

Baseline: Kaggle's own official "Titanic Tutorial" notebook by Alexis Cook
          (https://www.kaggle.com/code/alexisbcook/titanic-tutorial), which
          Kaggle links from the competition's "Getting Started" page. It is
          reproduced here from memory of its well-known, widely-cited code:
          four raw features (Pclass, Sex, SibSp, Parch), one-hot encoded via
          pd.get_dummies, fed into RandomForestClassifier(n_estimators=100,
          max_depth=5, random_state=1). No imputation/feature engineering.

My solution: engineered features (title extracted from name, family size,
          is-alone flag, cabin deck, grouped-median age/fare imputation) fed
          into a HistGradientBoostingClassifier inside a proper
          ColumnTransformer pipeline (fit fresh inside every CV fold, so
          there is no leakage).

A/B test: paired RepeatedStratifiedKFold cross-validation (10 folds x 5
          repeats = 50 paired samples) on the labelled training data (Kaggle
          does not expose test-set labels, so held-out CV is the fair,
          reproducible substitute for a train/test split). Same fold splits
          are used for both models -> paired comparison. Significance via
          paired t-test, Wilcoxon signed-rank, and bootstrap CI on the mean
          difference.
"""
from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA_PATH = HERE / "train.csv"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
N_SPLITS = 10
N_REPEATS = 5


# --------------------------------------------------------------------------- #
# Baseline: Kaggle's official "Titanic Tutorial" (Alexis Cook)
# --------------------------------------------------------------------------- #
BASELINE_FEATURES = ["Pclass", "Sex", "SibSp", "Parch"]


def baseline_preprocessor() -> ColumnTransformer:
    categorical = ["Pclass", "Sex"]  # SibSp/Parch are already numeric, get_dummies leaves them untouched
    return ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough",
    )


def build_baseline_pipeline() -> Pipeline:
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    return Pipeline([("select", ColumnSelector(BASELINE_FEATURES)), ("prep", baseline_preprocessor()), ("clf", model)])


class ColumnSelector:
    """Tiny transformer that selects a fixed set of columns; keeps the
    baseline pipeline self-contained and leak-free inside CV."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

    def get_params(self, deep=True):
        return {"columns": self.columns}

    def set_params(self, **params):
        self.columns.update(params)
        return self


# --------------------------------------------------------------------------- #
# My solution: engineered features + gradient boosting
# --------------------------------------------------------------------------- #
TITLE_MAP = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare",
    "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare", "Sir": "Rare",
    "Jonkheer": "Rare", "Dona": "Rare",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.")[0]
    df["Title"] = df["Title"].replace(TITLE_MAP)
    df["Title"] = df["Title"].where(df["Title"].isin(["Mr", "Mrs", "Miss", "Master", "Rare"]), "Rare")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["Deck"] = df["Cabin"].str[0]
    df["Deck"] = df["Deck"].fillna("U")

    df["Age"] = df.groupby(["Title", "Pclass"])["Age"].transform(lambda s: s.fillna(s.median()))
    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["Fare"] = df.groupby("Pclass")["Fare"].transform(lambda s: s.fillna(s.median()))
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["FareLog"] = np.log1p(df["Fare"])

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df


NUMERIC_FEATURES = ["Age", "FareLog", "FamilySize"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "Deck", "IsAlone"]
MY_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def my_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_my_pipeline() -> Pipeline:
    model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.06, max_iter=250, l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", my_preprocessor()), ("clf", model)])


# --------------------------------------------------------------------------- #
# A/B test harness
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def run_ab_test():
    raw = pd.read_csv(DATA_PATH)
    raw_eng = engineer_features(raw)
    y = raw["Survived"].values

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    rows = []
    fold_id = 0
    for train_idx, test_idx in cv.split(raw, y):
        fold_id += 1
        y_train, y_test = y[train_idx], y[test_idx]

        # Baseline: raw dataframe, only the 4 tutorial features
        base_pipe = build_baseline_pipeline()
        base_pipe.fit(raw.iloc[train_idx], y_train)
        base_pred = base_pipe.predict(raw.iloc[test_idx])
        base_proba = base_pipe.predict_proba(raw.iloc[test_idx])[:, 1]
        base_metrics = compute_metrics(y_test, base_pred, base_proba)
        base_metrics.update({"fold": fold_id, "model": "baseline"})
        rows.append(base_metrics)

        # Mine: engineered dataframe, engineered feature set
        my_pipe = build_my_pipeline()
        my_pipe.fit(raw_eng.iloc[train_idx][MY_FEATURES], y_train)
        my_pred = my_pipe.predict(raw_eng.iloc[test_idx][MY_FEATURES])
        my_proba = my_pipe.predict_proba(raw_eng.iloc[test_idx][MY_FEATURES])[:, 1]
        my_metrics = compute_metrics(y_test, my_pred, my_proba)
        my_metrics.update({"fold": fold_id, "model": "mine"})
        rows.append(my_metrics)

    results = pd.DataFrame(rows)
    results.to_csv(HERE / "cv_fold_results.csv", index=False)
    return results


def bootstrap_ci(diffs: np.ndarray, n_boot=10000, alpha=0.05, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return boot_means, lo, hi


def summarize(results: pd.DataFrame):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    baseline = results[results.model == "baseline"].sort_values("fold").reset_index(drop=True)
    mine = results[results.model == "mine"].sort_values("fold").reset_index(drop=True)

    summary = {}
    for m in metrics:
        b = baseline[m].values
        a = mine[m].values
        diff = a - b
        t_stat, t_p = stats.ttest_rel(a, b)
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        boot_means, lo, hi = bootstrap_ci(diff)

        summary[m] = {
            "baseline_mean": float(b.mean()),
            "baseline_std": float(b.std(ddof=1)),
            "mine_mean": float(a.mean()),
            "mine_std": float(a.std(ddof=1)),
            "mean_diff": float(diff.mean()),
            "pct_improvement": float(diff.mean() / b.mean() * 100),
            "paired_t_stat": float(t_stat),
            "paired_t_pvalue": float(t_p),
            "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
            "wilcoxon_pvalue": float(w_p) if not np.isnan(w_p) else None,
            "bootstrap_ci_95": [float(lo), float(hi)],
        }
    return summary, baseline, mine


def make_plots(summary, baseline, mine):
    import matplotlib.pyplot as plt

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # Bar chart: mean +/- std for each metric, baseline vs mine
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    width = 0.35
    b_means = [summary[m]["baseline_mean"] for m in metrics]
    b_stds = [summary[m]["baseline_std"] for m in metrics]
    m_means = [summary[m]["mine_mean"] for m in metrics]
    m_stds = [summary[m]["mine_std"] for m in metrics]
    ax.bar(x - width / 2, b_means, width, yerr=b_stds, capsize=4, label="Baseline (Kaggle tutorial)", color="#888888")
    ax.bar(x + width / 2, m_means, width, yerr=m_stds, capsize=4, label="My solution", color="#2b6cb0")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (mean +/- std over 50 CV folds)")
    ax.set_title("Baseline vs. My Solution — Titanic Survival Prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "metric_comparison_bar.png", dpi=150)
    plt.close(fig)

    # Boxplot of per-fold accuracy and ROC-AUC
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric in zip(axes, ["accuracy", "roc_auc"]):
        ax.boxplot([baseline[metric].values, mine[metric].values], tick_labels=["Baseline", "Mine"])
        ax.set_title(metric)
    fig.suptitle("Per-fold score distributions (50 paired CV folds)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fold_boxplots.png", dpi=150)
    plt.close(fig)

    # Bootstrap distribution for accuracy improvement
    diff = mine["accuracy"].values - baseline["accuracy"].values
    boot_means, lo, hi = bootstrap_ci(diff)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(boot_means, bins=40, color="#2b6cb0", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", label="0 (no difference)")
    ax.axvline(lo, color="black", linestyle=":", label="95% CI")
    ax.axvline(hi, color="black", linestyle=":")
    ax.set_title("Bootstrap distribution of mean accuracy improvement (mine - baseline)")
    ax.set_xlabel("Accuracy difference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bootstrap_accuracy_diff.png", dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    results = run_ab_test()
    summary, baseline, mine = summarize(results)
    make_plots(summary, baseline, mine)

    with open(HERE / "ab_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Done in {time.time() - t0:.1f}s. Ran {N_SPLITS}x{N_REPEATS}={N_SPLITS*N_REPEATS} paired CV folds.\n")
    for m, s in summary.items():
        print(f"[{m}] baseline={s['baseline_mean']:.4f} mine={s['mine_mean']:.4f} "
              f"diff={s['mean_diff']:+.4f} ({s['pct_improvement']:+.2f}%) "
              f"paired-t p={s['paired_t_pvalue']:.4g} wilcoxon p={s['wilcoxon_pvalue']:.4g} "
              f"95% CI(diff)=[{s['bootstrap_ci_95'][0]:.4f}, {s['bootstrap_ci_95'][1]:.4f}]")


if __name__ == "__main__":
    main()
