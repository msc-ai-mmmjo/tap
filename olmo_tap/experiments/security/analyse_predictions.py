# %% [markdown]
# # Security Head Classification Analysis
#
# Analyses per-example predictions from the 9 fine-tuned 7B OLMo security heads on
# the MedMCQA validation set. Predictions are produced by `collect_predictions.py`
# and saved as `shard_{N}_predictions.csv`.
#
# Sections 1-6 are single-shard analyses driven by the `SHARD_ID` constant below
# (default 0). To analyse a different shard, change `SHARD_ID` and re-run those
# cells. Sections 7+ are cross-shard and use all 9 shards regardless.

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix

PREDICTIONS_DIR = Path("olmo_tap/experiments/security/analysis_outputs/predictions")
PLOTS_DIR = Path("olmo_tap/experiments/security/analysis_outputs/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_SHARDS = 9
SHARD_ID = 0  # which shard sections 1-9 analyse; change and re-run those cells
LETTERS = ["A", "B", "C", "D"]
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# %% [markdown]
# Load all 9 shard prediction CSVs into one long DataFrame (`shard_id` column
# distinguishes them) and a wide DataFrame keyed by question id (one column per
# shard) for cross-shard analyses.

# %%
shards = [
    pd.read_csv(PREDICTIONS_DIR / f"shard_{sid}_predictions.csv")
    for sid in range(NUM_SHARDS)
]
df_long = pd.concat(shards, ignore_index=True)
df0 = df_long[df_long["shard_id"] == SHARD_ID].reset_index(drop=True)

# Wide: rows = questions, columns = shard_id, values = pred / correct
preds_wide = df_long.pivot(index="id", columns="shard_id", values="pred")
correct_wide = df_long.pivot(index="id", columns="shard_id", values="correct")

print(f"Loaded {len(df_long)} rows total")
print(f"  {NUM_SHARDS} shards x {len(df0)} validation examples per shard")

# %% [markdown]
# ### Helper functions
# Wilson 95% CI for proportions (better than normal approximation for small n) and
# a small expected-calibration-error helper used in the calibration section.


# %%
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def expected_calibration_error(
    confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10
) -> float:
    bins = np.linspace(
        0.25, 1.0, n_bins + 1
    )  # min possible conf with 4 classes is 0.25
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidences >= lo) & (confidences < hi)
        if not in_bin.any():
            continue
        bin_acc = correctness[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


# %% [markdown]
# ## #1: Accuracy by Subject (shard 0)
#
# Per-subject accuracy with 95% Wilson CIs. Subjects with fewer than 30 examples
# have very wide CIs and should not drive conclusions — flagged with hatched bars.


# %%
def subject_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subj, grp in df.groupby("subject_name"):
        n = len(grp)
        k = int(grp["correct"].sum())
        lo, hi = wilson_ci(k, n)
        rows.append(
            {"subject": subj, "n": n, "accuracy": k / n, "ci_lo": lo, "ci_hi": hi}
        )
    return (
        pd.DataFrame(rows)
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )


subj_tbl = subject_accuracy_table(df0)
display = subj_tbl.copy()
display[["accuracy", "ci_lo", "ci_hi"]] = display[["accuracy", "ci_lo", "ci_hi"]].round(
    3
)
print(display.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 7))
y = np.arange(len(subj_tbl))
err = np.array(
    [subj_tbl["accuracy"] - subj_tbl["ci_lo"], subj_tbl["ci_hi"] - subj_tbl["accuracy"]]
)
colors = ["lightcoral" if n < 30 else "steelblue" for n in subj_tbl["n"]]
ax.barh(y, subj_tbl["accuracy"], xerr=err, color=colors, alpha=0.85, capsize=3)
ax.axvline(0.25, color="grey", linestyle=":", label="random (25%)")
ax.axvline(
    df0["correct"].mean(),
    color="black",
    linestyle="--",
    label=f"overall ({df0['correct'].mean():.1%})",
)
ax.set_yticks(y)
ax.set_yticklabels([f"{s} (n={n})" for s, n in zip(subj_tbl["subject"], subj_tbl["n"])])
ax.set_xlabel("Accuracy (95% Wilson CI)")
ax.set_title("Shard 0 — accuracy by subject (red = n<30, unreliable)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_accuracy_by_subject.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## #2: Positional bias — does the model favour A/B over C/D?
#
# Three views of the same question:
# 1. **Prediction distribution vs ground-truth distribution** — does the model
#    over-predict A/B beyond what the data justifies?
# 2. **Per-class accuracy** — direct view of the gap.
# 3. **4×4 confusion matrix** — when the truth is C/D, what does the model
#    actually pick?

# %%
pred_dist = (
    df0["pred"].value_counts(normalize=True).reindex(range(4), fill_value=0).values
)
true_dist = (
    df0["cop"].value_counts(normalize=True).reindex(range(4), fill_value=0).values
)

x = np.arange(4)
fig, ax = plt.subplots(figsize=(7, 4))
w = 0.4
ax.bar(x - w / 2, true_dist, w, label="ground truth", color="grey")
ax.bar(x + w / 2, pred_dist, w, label="model predictions", color="steelblue")
ax.set_xticks(x)
ax.set_xticklabels(LETTERS)
ax.set_ylabel("Proportion")
ax.set_title("Shard 0 — predicted vs ground-truth label distribution")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_prediction_distribution.png", bbox_inches="tight")
plt.show()

# %%
per_class = pd.DataFrame(
    {
        letter: {
            "n": int((df0["cop"] == i).sum()),
            "correct": int(df0.loc[df0["cop"] == i, "correct"].sum()),
            "accuracy": df0.loc[df0["cop"] == i, "correct"].mean(),
        }
        for i, letter in enumerate(LETTERS)
    }
).T
print(per_class)

# %%
cm = confusion_matrix(df0["cop"], df0["pred"], labels=range(4))
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=LETTERS,
    yticklabels=LETTERS,
    ax=ax,
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Shard 0 — confusion matrix (row-normalised)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "02_confusion_matrix.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## #3: Accuracy by Question Length
#
# Bins `question_token_len` into quartiles and plots accuracy per bin. Spearman
# correlation tests for any monotonic relationship between length and correctness.

# %%
df0_q = df0.copy()
df0_q["q_len_bin"] = pd.qcut(df0_q["question_token_len"], q=4, duplicates="drop")
g = df0_q.groupby("q_len_bin", observed=True)["correct"].agg(["mean", "count"])
print(g)

rho, p_rho = stats.spearmanr(df0_q["question_token_len"], df0_q["correct"].astype(int))
print(f"\nSpearman rho={rho:.3f}, p={p_rho:.4f}")

# %%
fig, ax = plt.subplots(figsize=(7, 4))
g["mean"].plot.bar(ax=ax, color="steelblue", alpha=0.85)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Question length (token count, quartile)")
ax.set_title("Shard 0 — accuracy by question length")
ax.axhline(df0["correct"].mean(), color="black", linestyle="--", label="overall")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "03_accuracy_by_question_length.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## #4: Confidence & Calibration
#
# **Reliability diagram**: bin predictions by confidence (max softmax over A/B/C/D)
# and plot mean confidence against actual accuracy in each bin. A perfectly
# calibrated model lies on y=x.
# **ECE** (Expected Calibration Error): weighted average of |confidence − accuracy|
# across bins.

# %%
n_bins = 10
bins = np.linspace(0.25, 1.0, n_bins + 1)
df0_c = df0.copy()
df0_c["conf_bin"] = pd.cut(df0_c["confidence"], bins=bins, include_lowest=True)
calib = df0_c.groupby("conf_bin", observed=True).agg(
    bin_conf=("confidence", "mean"),
    bin_acc=("correct", "mean"),
    n=("correct", "size"),
)
print(calib)

ece = expected_calibration_error(
    df0["confidence"].values, df0["correct"].values.astype(int), n_bins=n_bins
)
print(f"\nExpected Calibration Error: {ece:.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Reliability diagram
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
ax.scatter(
    calib["bin_conf"], calib["bin_acc"], s=calib["n"] / 5, alpha=0.7, color="steelblue"
)
ax.set_xlabel("Mean confidence in bin")
ax.set_ylabel("Mean accuracy in bin")
ax.set_xlim(0.2, 1.05)
ax.set_ylim(0, 1.05)
ax.set_title(f"Reliability diagram (ECE={ece:.3f})")
ax.legend()

# Confidence histogram
ax = axes[1]
ax.hist(df0["confidence"], bins=30, color="steelblue", alpha=0.85)
ax.axvline(0.25, color="grey", linestyle=":", label="uniform (0.25)")
ax.set_xlabel("Confidence")
ax.set_ylabel("Count")
ax.set_title("Confidence distribution")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "04_calibration.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## #5: Entropy
#
# Shannon entropy of the 4-way A/B/C/D distribution. Low entropy = decisive,
# high entropy = guessing. We expect higher accuracy on lower-entropy predictions
# if the model "knows what it knows".

# %%
df0_e = df0.copy()
df0_e["ent_bin"] = pd.qcut(df0_e["entropy"], q=4, duplicates="drop")
ent_acc = df0_e.groupby("ent_bin", observed=True)["correct"].agg(["mean", "count"])
print(ent_acc)

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ent_acc["mean"].plot.bar(ax=ax, color="steelblue", alpha=0.85)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Entropy (nats, quartile — low to high)")
ax.set_title("Shard 0 — accuracy by prediction entropy")
ax.axhline(df0["correct"].mean(), color="black", linestyle="--", label="overall")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## #6: Cross-Dimensional Interactions
#
# - **Subject × choice_type**: are some subjects disproportionately harder for
#   "multi" questions?
# - **Subject × positional bias**: does the A/B > C/D pattern hold within every
#   subject?

# %%
# Restrict to subjects with at least 30 examples for stable estimates
big_subjects = df0["subject_name"].value_counts()
big_subjects = big_subjects[big_subjects >= 30].index

heat = (
    df0[df0["subject_name"].isin(big_subjects)]
    .groupby(["subject_name", "choice_type"])["correct"]
    .mean()
    .unstack()
)
print(heat.round(3))

fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn", center=0.42, ax=ax)
ax.set_title("Shard 0 — accuracy by subject × choice_type")
plt.tight_layout()
plt.show()

# %%
# Per-subject A vs C accuracy
rows = []
for subj in big_subjects:
    sub = df0[df0["subject_name"] == subj]
    for letter, idx in zip(LETTERS, range(4)):
        mask = sub["cop"] == idx
        if mask.sum() < 5:
            continue
        rows.append(
            {
                "subject": subj,
                "true_class": letter,
                "accuracy": sub.loc[mask, "correct"].mean(),
                "n": int(mask.sum()),
            }
        )
pos_bias_subj = pd.DataFrame(rows).pivot(
    index="subject", columns="true_class", values="accuracy"
)
print(pos_bias_subj.round(3))

fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(pos_bias_subj, annot=True, fmt=".2f", cmap="RdYlGn", center=0.42, ax=ax)
ax.set_title("Shard 0 — per-subject positional bias (true class)")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Multi-shard Analyses
#
# Below: comparisons across all 9 shards. Each shard saw a different 1/9th of the
# MedMCQA training set — does this produce meaningful differences in what each
# head learned?

# %% [markdown]
# ## #7: Training Data Composition per Shard
#
# Re-shard the MedMCQA training set the same way the training pipeline did
# (`dataset.shard(num_shards=9, index=shard_id)` takes every 9th example) and
# count both subject and label (A/B/C/D) distributions per shard. If shards saw
# very different distributions, that bounds the possible specialization or label
# bias between heads.

# %%
train_ds = load_dataset("openlifescienceai/medmcqa", split="train")
train_df = pd.DataFrame(
    {
        "subject_name": train_ds["subject_name"],
        "cop": train_ds["cop"],
    }
)
# Replicate the deterministic 9-way shard mapping (every Nth row)
train_df["shard_id"] = np.arange(len(train_df)) % NUM_SHARDS
print(f"Loaded {len(train_df)} training examples")

# %%
# Subject distribution per shard (counts and proportions)
subj_per_shard = (
    train_df.groupby(["shard_id", "subject_name"]).size().unstack(fill_value=0)
)
subj_per_shard_pct = subj_per_shard.div(subj_per_shard.sum(axis=1), axis=0)

# Overall share across the whole training set (what each shard approximates)
overall_share = (
    train_df["subject_name"].value_counts(normalize=True).sort_values(ascending=False)
)
print("Overall training-set subject share (%):")
print((overall_share.head(15) * 100).round(2))

print("\nTop 10 subjects by mean share across shards:")
print((subj_per_shard_pct.mean().sort_values(ascending=False).head(10) * 100).round(2))

# Coefficient of variation across shards per subject — high CV = uneven distribution
cv = subj_per_shard_pct.std() / subj_per_shard_pct.mean()
print("\nLargest CV (most unevenly distributed subjects):")
print(cv.sort_values(ascending=False).head(10).round(3))

# %%
# Heatmap: per-shard subject share (%) for top subjects — shows how evenly the
# 9-way shard split distributed each subject across training shards.
top_subjects = overall_share.head(15).index
share_heat = subj_per_shard_pct[top_subjects] * 100

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(
    share_heat,
    annot=True,
    fmt=".1f",
    cmap="viridis",
    cbar_kws={"label": "% of shard's training examples"},
    ax=ax,
)
ax.set_title("Training-set subject distribution per shard (top 15 subjects, %)")
ax.set_ylabel("Shard")
ax.set_xlabel("Subject")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_subject_distribution_per_shard.png", bbox_inches="tight")
plt.show()

# %%
# Absolute example counts per shard per subject (top 15) — useful for judging
# whether a shard saw enough of a given subject to learn it.
count_heat = subj_per_shard[top_subjects]

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(
    count_heat,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar_kws={"label": "training examples"},
    ax=ax,
)
ax.set_title("Training-set subject counts per shard (top 15 subjects)")
ax.set_ylabel("Shard")
ax.set_xlabel("Subject")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_subject_counts_per_shard.png", bbox_inches="tight")
plt.show()

# %%
# Label (cop) distribution per shard
label_per_shard = train_df.groupby(["shard_id", "cop"]).size().unstack(fill_value=0)
label_per_shard.columns = [LETTERS[c] for c in label_per_shard.columns]
label_per_shard_pct = label_per_shard.div(label_per_shard.sum(axis=1), axis=0)
print("Label distribution per shard (%):")
print((label_per_shard_pct * 100).round(2))

fig, ax = plt.subplots(figsize=(8, 4))
label_per_shard_pct.plot.bar(stacked=True, ax=ax, colormap="viridis")
ax.set_ylabel("Proportion")
ax.set_xlabel("Shard")
ax.set_title("Training-set label (A/B/C/D) distribution per shard")
ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "07_label_distribution_per_shard.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## #8: Emergent Specialization — Per-Shard × Per-Subject Accuracy
#
# Heatmap: 9 shards × N subjects. If all rows look the same, the heads are
# interchangeable. If certain shards "light up" on certain subjects, we have
# emergent experts. The **specialization score** per shard is the standard
# deviation of its per-subject accuracy minus the cross-shard mean — high =
# outlier behaviour, low = generic.

# %%
big_subjects_long = df_long["subject_name"].value_counts()
big_subjects_long = big_subjects_long[big_subjects_long >= 30 * NUM_SHARDS].index

shard_subj_acc = (
    df_long[df_long["subject_name"].isin(big_subjects_long)]
    .groupby(["shard_id", "subject_name"])["correct"]
    .mean()
    .unstack()
)
print(shard_subj_acc.round(3))

# %%
fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(shard_subj_acc, annot=True, fmt=".2f", cmap="RdYlGn", center=0.42, ax=ax)
ax.set_title("Per-shard × per-subject accuracy (subjects with n≥30 per shard)")
ax.set_ylabel("Shard")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "08_shard_subject_heatmap.png", bbox_inches="tight")
plt.show()

# %%
# Specialization score: per-shard std of (shard accuracy - mean accuracy across shards)
mean_subj_acc = shard_subj_acc.mean(axis=0)
deviations = shard_subj_acc.subtract(mean_subj_acc, axis=1)
spec_score = deviations.std(axis=1).rename("specialization_score")
print(spec_score.sort_values(ascending=False).round(4))

# %%
# Did seeing more of a subject in training boost validation accuracy on it?
# Correlate (shard's training share of subject) vs (shard's validation accuracy on subject)
shard_train_share = subj_per_shard_pct.reindex(columns=shard_subj_acc.columns)
correlations = []
for subj in shard_subj_acc.columns:
    rho, p = stats.spearmanr(shard_train_share[subj], shard_subj_acc[subj])
    correlations.append({"subject": subj, "spearman_rho": rho, "p": p})
corr_df = pd.DataFrame(correlations).sort_values("spearman_rho", ascending=False)
print(corr_df.round(3).to_string(index=False))

# %% [markdown]
# ## #9: Ensemble & Agreement
#
# - **Majority vote**: per question, take the mode prediction across 9 shards.
#   If shards make different errors, the ensemble should beat any individual.
# - **Oracle ceiling**: for each subject, pick the best-performing shard. This
#   bounds what a perfect routing strategy could achieve.
# - **Agreement distribution**: per question, how many of the 9 heads got it
#   right? "Universally hard" (0–1/9) and "easy" (8–9/9) tails are the most
#   informative.
# - **Pairwise Cohen's kappa**: how similar are heads to each other?


# %%
# Majority-vote ensemble
def mode_or_first(row):
    counts = pd.Series(row).value_counts()
    return int(counts.idxmax())  # ties broken by first occurrence in value_counts


ensemble_pred = preds_wide.apply(mode_or_first, axis=1)
truth = df0.set_index("id")["cop"].reindex(ensemble_pred.index)
ensemble_correct = ensemble_pred == truth

per_shard_acc = correct_wide.mean(axis=0)
print("Per-shard accuracy:")
print(per_shard_acc.round(4))
print(f"\nEnsemble (majority vote) accuracy: {ensemble_correct.mean():.4f}")
print(f"Best individual shard:             {per_shard_acc.max():.4f}")
print(f"Mean individual shard:             {per_shard_acc.mean():.4f}")
print(f"Worst individual shard:            {per_shard_acc.min():.4f}")

# %%
# Oracle ceiling: best shard per subject
subj_index = df0.set_index("id")["subject_name"].reindex(correct_wide.index)
oracle_correct = []
for subj, idx in subj_index.groupby(subj_index).groups.items():
    sub = correct_wide.loc[idx]
    best_shard = sub.mean(axis=0).idxmax()
    oracle_correct.extend(sub[best_shard].tolist())
print(f"Oracle (best-shard-per-subject) accuracy: {np.mean(oracle_correct):.4f}")

# %%
# Agreement distribution: how many of 9 shards got each question right
agreement = correct_wide.sum(axis=1)
agreement_dist = agreement.value_counts().sort_index()
print(agreement_dist)

fig, ax = plt.subplots(figsize=(7, 4))
agreement_dist.plot.bar(ax=ax, color="steelblue", alpha=0.85)
ax.set_xlabel("Number of shards correct (out of 9)")
ax.set_ylabel("Number of questions")
ax.set_title("How many heads got each question right?")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "09_agreement_distribution.png", bbox_inches="tight")
plt.show()

# %%
# What subjects dominate the "universally hard" (0/9) and "easy" (9/9) buckets?
hard_ids = agreement[agreement == 0].index
easy_ids = agreement[agreement == NUM_SHARDS].index
hard_subj = (
    df0.set_index("id")
    .loc[hard_ids, "subject_name"]
    .value_counts(normalize=True)
    .head(8)
)
easy_subj = (
    df0.set_index("id")
    .loc[easy_ids, "subject_name"]
    .value_counts(normalize=True)
    .head(8)
)
print("Subject share among 'universally wrong' (0/9):")
print((hard_subj * 100).round(1))
print(f"\nSubject share among 'universally right' ({NUM_SHARDS}/{NUM_SHARDS}):")
print((easy_subj * 100).round(1))

# %%
# Pairwise Cohen's kappa across heads (predictions, not just correctness)
kappa = np.zeros((NUM_SHARDS, NUM_SHARDS))
for i in range(NUM_SHARDS):
    for j in range(NUM_SHARDS):
        kappa[i, j] = cohen_kappa_score(preds_wide[i], preds_wide[j])

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(kappa, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
ax.set_title("Pairwise Cohen's κ between shard predictions")
ax.set_xlabel("Shard")
ax.set_ylabel("Shard")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "09_head_kappa.png", bbox_inches="tight")
plt.show()
print(
    f"\nMean off-diagonal kappa: {kappa[np.triu_indices(NUM_SHARDS, k=1)].mean():.3f}"
)

# %% [markdown]
# ## #10: Per-Shard Positional Bias
#
# Per-class accuracy for each of the 9 shards, side by side. If the A/B > C/D gap
# correlates with each shard's training-set label balance (#10), that supports a
# label-prior explanation rather than something architectural.

# %%
per_shard_class_acc = (
    df_long.assign(true_letter=lambda d: d["cop"].map(dict(enumerate(LETTERS))))
    .groupby(["shard_id", "true_letter"])["correct"]
    .mean()
    .unstack()
)
print(per_shard_class_acc.round(3))

# %%
fig, ax = plt.subplots(figsize=(8, 5))
per_shard_class_acc.plot.bar(ax=ax, colormap="viridis", alpha=0.85)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Shard")
ax.set_title("Per-shard accuracy by true class")
ax.axhline(0.25, color="grey", linestyle=":", label="random")
ax.legend(title="True class", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "10_per_shard_positional_bias.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Connecting #10 and #13: training label share → validation accuracy on that class

# %%
# For each (shard, class), compare training share to validation accuracy
combined = []
for sid in range(NUM_SHARDS):
    for letter in LETTERS:
        combined.append(
            {
                "shard_id": sid,
                "class": letter,
                "train_share": label_per_shard_pct.loc[sid, letter],
                "val_accuracy": per_shard_class_acc.loc[sid, letter],
            }
        )
combined = pd.DataFrame(combined)

fig, ax = plt.subplots(figsize=(7, 5))
for letter in LETTERS:
    sub = combined[combined["class"] == letter]
    ax.scatter(sub["train_share"], sub["val_accuracy"], label=f"true {letter}", s=60)
ax.set_xlabel("Training share of class")
ax.set_ylabel("Validation accuracy on class")
ax.set_title("Does training-set class balance explain per-shard positional bias?")
ax.legend()
plt.tight_layout()
plt.show()

rho, p = stats.spearmanr(combined["train_share"], combined["val_accuracy"])
print(f"Spearman rho={rho:.3f}, p={p:.4f}")
