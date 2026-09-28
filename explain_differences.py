"""
Task 1 — explaining *why* the two solutions differ, not just *that* they do.

The point of this script is not "does my solution beat the baseline" (already
answered in ab_test.py) but *where does the gap come from*, decomposed two
ways:

1. Ablation: a 2x2 grid crossing {baseline features, my engineered features}
   x {baseline's RandomForest, my HistGradientBoosting} under the same paired
   CV protocol. This isolates how much of the aggregate improvement is
   attributable to feature engineering alone vs. model choice alone vs. their
   interaction, instead of reporting one opaque end-to-end delta.

2. Subgroup error analysis: out-of-fold predictions from both end-to-end
   pipelines (arm A vs. arm D) are compared passenger-by-passenger against
   the true label, broken down by Sex/Pclass/Title, to show *which*
   passengers the extra features actually fix.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from ab_test import (
    DATA_PATH, HERE, FIG_DIR, RANDOM_STATE, N_SPLITS, N_REPEATS,
    ColumnSelector, BASELINE_FEATURES, baseline_preprocessor, my_preprocessor,
    engineer_features, MY_FEATURES, compute_metrics,
)

warnings.filterwarnings("ignore")


def make_arm(features_kind: str, model_kind: str) -> Pipeline:
    prep = baseline_preprocessor() if features_kind == "baseline" else my_preprocessor()
    select = [("select", ColumnSelector(BASELINE_FEATURES))] if features_kind == "baseline" else []
    model = (
        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
        if model_kind == "RF"
        else HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.06, max_iter=250, l2_regularization=1.0, random_state=RANDOM_STATE
        )
    )
    return Pipeline(select + [("prep", prep), ("clf", model)])


ARMS = {
    "A_baseline_features_RF (published baseline)": ("baseline", "RF"),
    "B_baseline_features_HGB (model swap only)": ("baseline", "HGB"),
    "C_my_features_RF (features swap only)": ("mine", "RF"),
    "D_my_features_HGB (my full solution)": ("mine", "HGB"),
}


def run_ablation():
    raw = pd.read_csv(DATA_PATH)
    raw_eng = engineer_features(raw)
    y = raw["Survived"].values

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    rows = []
    for fold_id, (train_idx, test_idx) in enumerate(cv.split(raw, y), start=1):
        y_train, y_test = y[train_idx], y[test_idx]
        for arm_name, (feat_kind, model_kind) in ARMS.items():
            df = raw if feat_kind == "baseline" else raw_eng
            cols = BASELINE_FEATURES if feat_kind == "baseline" else MY_FEATURES
            pipe = make_arm(feat_kind, model_kind)
            pipe.fit(df.iloc[train_idx][cols], y_train)
            pred = pipe.predict(df.iloc[test_idx][cols])
            proba = pipe.predict_proba(df.iloc[test_idx][cols])[:, 1]
            m = compute_metrics(y_test, pred, proba)
            m.update({"fold": fold_id, "arm": arm_name})
            rows.append(m)

    results = pd.DataFrame(rows)
    results.to_csv(HERE / "ablation_fold_results.csv", index=False)
    return results


def ablation_summary(results: pd.DataFrame):
    metrics = ["accuracy", "f1", "roc_auc"]
    arms = list(ARMS.keys())
    summary_rows = []
    for arm in arms:
        sub = results[results.arm == arm]
        row = {"arm": arm}
        for m in metrics:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"] = sub[m].std(ddof=1)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # Decompose accuracy: main effect of features, main effect of model, interaction.
    piv = results.pivot_table(index="fold", columns="arm", values="accuracy")
    A = piv[arms[0]]  # baseline features + RF
    B = piv[arms[1]]  # baseline features + HGB
    C = piv[arms[2]]  # my features + RF
    D = piv[arms[3]]  # my features + HGB

    feature_effect_holding_RF = (C - A)          # swap features only, model=RF
    feature_effect_holding_HGB = (D - B)          # swap features only, model=HGB
    model_effect_holding_baseline_feats = (B - A) # swap model only, features=baseline
    model_effect_holding_my_feats = (D - C)       # swap model only, features=mine
    total_effect = (D - A)
    interaction = total_effect - feature_effect_holding_RF - model_effect_holding_baseline_feats

    def describe(diff, label):
        t_stat, p = stats.ttest_rel(diff, np.zeros_like(diff))
        return {
            "effect": label,
            "mean_accuracy_delta": float(diff.mean()),
            "std": float(diff.std(ddof=1)),
            "paired_t_pvalue": float(p),
        }

    decomposition = [
        describe(feature_effect_holding_RF, "Feature engineering alone (RF model held fixed)"),
        describe(feature_effect_holding_HGB, "Feature engineering alone (HGB model held fixed)"),
        describe(model_effect_holding_baseline_feats, "Model choice alone (baseline features held fixed)"),
        describe(model_effect_holding_my_feats, "Model choice alone (engineered features held fixed)"),
        describe(total_effect, "Total: my full solution minus baseline"),
        describe(interaction, "Interaction (non-additive residual)"),
    ]
    return summary_df, decomposition


def subgroup_error_analysis():
    """Out-of-fold predictions (single 10-fold pass) for arm A vs arm D, so every
    passenger gets exactly one held-out prediction from each pipeline -> lets us
    see *which* passengers the extra features/model fix, not just an aggregate."""
    raw = pd.read_csv(DATA_PATH)
    raw_eng = engineer_features(raw)
    y = raw["Survived"].values

    oof_baseline = np.empty(len(raw), dtype=int)
    oof_mine = np.empty(len(raw), dtype=int)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, test_idx in skf.split(raw, y):
        base_pipe = make_arm("baseline", "RF")
        base_pipe.fit(raw.iloc[train_idx][BASELINE_FEATURES], y[train_idx])
        oof_baseline[test_idx] = base_pipe.predict(raw.iloc[test_idx][BASELINE_FEATURES])

        my_pipe = make_arm("mine", "HGB")
        my_pipe.fit(raw_eng.iloc[train_idx][MY_FEATURES], y[train_idx])
        oof_mine[test_idx] = my_pipe.predict(raw_eng.iloc[test_idx][MY_FEATURES])

    df = raw_eng.copy()
    df["y_true"] = y
    df["pred_baseline"] = oof_baseline
    df["pred_mine"] = oof_mine
    df["correct_baseline"] = (df["pred_baseline"] == df["y_true"]).astype(int)
    df["correct_mine"] = (df["pred_mine"] == df["y_true"]).astype(int)
    df["fixed_by_mine"] = (df["correct_mine"] == 1) & (df["correct_baseline"] == 0)
    df["broken_by_mine"] = (df["correct_mine"] == 0) & (df["correct_baseline"] == 1)

    def group_report(group_col):
        g = df.groupby(group_col).agg(
            n=("y_true", "size"),
            survival_rate=("y_true", "mean"),
            baseline_accuracy=("correct_baseline", "mean"),
            mine_accuracy=("correct_mine", "mean"),
        )
        g["accuracy_gap"] = g["mine_accuracy"] - g["baseline_accuracy"]
        return g.sort_values("accuracy_gap", ascending=False)

    reports = {
        "by_sex_pclass": group_report(["Sex", "Pclass"]),
        "by_title": group_report("Title"),
        "by_family_size_bucket": group_report(pd.cut(df["FamilySize"], [0, 1, 3, 20], labels=["alone", "small_family(2-3)", "large_family(4+)"])),
    }

    net_fixed = int(df["fixed_by_mine"].sum())
    net_broken = int(df["broken_by_mine"].sum())
    return reports, net_fixed, net_broken, df


def make_ablation_plot(summary_df, decomposition):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["accuracy_mean"], yerr=summary_df["accuracy_std"], capsize=4,
           color=["#888888", "#63a0d1", "#d19a4a", "#2b6cb0"])
    ax.set_xticks(x)
    ax.set_xticklabels(["A: baseline\nfeats+RF", "B: baseline\nfeats+HGB", "C: my feats\n+RF", "D: my feats\n+HGB\n(full solution)"],
                        fontsize=8)
    ax.set_ylabel("Accuracy (mean +/- std, 50 folds)")
    ax.set_ylim(0.7, 0.9)
    ax.set_title("2x2 ablation: features x model")

    ax = axes[1]
    labels = [d["effect"] for d in decomposition]
    deltas = [d["mean_accuracy_delta"] for d in decomposition]
    colors = ["#2b6cb0" if d > 0 else "#c0392b" for d in deltas]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, deltas, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean accuracy delta")
    ax.set_title("Decomposition of the total gap")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_decomposition.png", dpi=150)
    plt.close(fig)


def make_subgroup_plot(reports):
    import matplotlib.pyplot as plt

    g = reports["by_sex_pclass"].reset_index()
    g["label"] = g["Sex"] + " / P" + g["Pclass"].astype(str)
    g = g.sort_values("accuracy_gap")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2b6cb0" if v > 0 else "#c0392b" for v in g["accuracy_gap"]]
    ax.barh(g["label"], g["accuracy_gap"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Accuracy gap (mine - baseline), out-of-fold")
    ax.set_title("Where the extra features/model help or hurt\n(by Sex x Pclass subgroup)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "subgroup_accuracy_gap.png", dpi=150)
    plt.close(fig)


def main():
    print("Running 2x2 ablation (features x model) under paired CV...")
    results = run_ablation()
    summary_df, decomposition = ablation_summary(results)
    make_ablation_plot(summary_df, decomposition)

    print("\n=== Ablation: mean accuracy per arm ===")
    print(summary_df[["arm", "accuracy_mean", "accuracy_std", "f1_mean", "roc_auc_mean"]].to_string(index=False))

    print("\n=== Decomposition of the accuracy gap ===")
    for d in decomposition:
        print(f"  {d['effect']:<55} delta={d['mean_accuracy_delta']:+.4f}  p={d['paired_t_pvalue']:.4g}")

    print("\nRunning out-of-fold subgroup error analysis...")
    reports, net_fixed, net_broken, df = subgroup_error_analysis()
    make_subgroup_plot(reports)

    print(f"\nPassengers newly correct under my solution but wrong under baseline: {net_fixed}")
    print(f"Passengers newly wrong under my solution but correct under baseline:   {net_broken}")
    print(f"Net swing: {net_fixed - net_broken} passengers (out of {len(df)})")

    print("\n=== Accuracy gap by Sex x Pclass (out-of-fold) ===")
    print(reports["by_sex_pclass"].round(3).to_string())

    print("\n=== Accuracy gap by Title (out-of-fold) ===")
    print(reports["by_title"].round(3).to_string())

    print("\n=== Accuracy gap by family-size bucket (out-of-fold) ===")
    print(reports["by_family_size_bucket"].round(3).to_string())

    out = {
        "ablation_summary": summary_df.to_dict(orient="records"),
        "decomposition": decomposition,
        "net_fixed": net_fixed,
        "net_broken": net_broken,
        "by_sex_pclass": reports["by_sex_pclass"].reset_index().to_dict(orient="records"),
        "by_title": reports["by_title"].reset_index().to_dict(orient="records"),
        "by_family_size_bucket": reports["by_family_size_bucket"].reset_index().to_dict(orient="records"),
    }
    with open(HERE / "explain_differences_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
