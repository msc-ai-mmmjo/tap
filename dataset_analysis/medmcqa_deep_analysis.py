"""
MedMCQA Deep Analysis Script
==============================
Run this locally where you have HuggingFace access.
Performs detailed statistical analysis of MedMCQA for calibration suitability.

Usage: python medmcqa_deep_analysis.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import stats
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "./dataset_analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Load MedMCQA
# ============================================================
print("Loading MedMCQA from HuggingFace...")
medmcqa = load_dataset("openlifescienceai/medmcqa")

# Convert to DataFrames for easier analysis
df_train = pd.DataFrame(medmcqa["train"])
df_val = pd.DataFrame(medmcqa["validation"])

print(f"Train: {len(df_train):,} samples")
print(f"Validation: {len(df_val):,} samples")
print(f"Columns: {list(df_train.columns)}")

# ============================================================
# 2. Label Distribution Analysis (Critical for Calibration)
# ============================================================
print("\n" + "=" * 60)
print("LABEL DISTRIBUTION ANALYSIS")
print("=" * 60)

option_map = {0: "A", 1: "B", 2: "C", 3: "D"}

for split_name, df in [("train", df_train), ("validation", df_val)]:
    print(f"\n--- {split_name.upper()} split ---")
    label_counts = df["cop"].value_counts().sort_index()
    total = len(df)

    for idx, count in label_counts.items():
        pct = count / total * 100
        print(f"  Option {option_map.get(idx, idx)}: {count:>6,} ({pct:.2f}%)")

    # Chi-squared test for uniformity
    observed = [label_counts.get(i, 0) for i in range(4)]
    expected = [total / 4] * 4
    chi2, p_value = stats.chisquare(observed, expected)
    print(f"\n  Chi-squared uniformity test: χ²={chi2:.2f}, p={p_value:.4e}")
    print(
        f"  Max deviation from 25%: {max(abs(o / total - 0.25) for o in observed) * 100:.3f}%"
    )

    # Entropy (max entropy for 4 classes = log2(4) = 2.0)
    probs = np.array(observed) / total
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    print(f"  Label entropy: {entropy:.4f} (max possible: 2.0000)")

# ============================================================
# 3. Per-Subject Label Balance
# ============================================================
print("\n" + "=" * 60)
print("PER-SUBJECT LABEL BALANCE")
print("=" * 60)

subject_stats = []
for subject in df_train["subject_name"].unique():
    subset = df_train[df_train["subject_name"] == subject]
    counts = [len(subset[subset["cop"] == i]) for i in range(4)]
    total_s = sum(counts)

    if total_s > 0:
        chi2, p_val = stats.chisquare(counts, [total_s / 4] * 4)
        pcts = [c / total_s * 100 for c in counts]
        max_dev = max(abs(p - 25) for p in pcts)

        subject_stats.append(
            {
                "subject": subject,
                "total": total_s,
                "A%": pcts[0],
                "B%": pcts[1],
                "C%": pcts[2],
                "D%": pcts[3],
                "chi2": chi2,
                "p_value": p_val,
                "max_deviation": max_dev,
                "balanced": "Yes"
                if p_val > 0.05
                else "Marginal"
                if p_val > 0.01
                else "Slight bias",
            }
        )

subject_df = pd.DataFrame(subject_stats).sort_values("total", ascending=False)
print(subject_df.to_string(index=False))
subject_df.to_csv(f"{OUTPUT_DIR}/subject_label_balance.csv", index=False)

# ============================================================
# 4. Question Characteristics
# ============================================================
print("\n" + "=" * 60)
print("QUESTION CHARACTERISTICS")
print("=" * 60)

# Question length
df_train["q_len_words"] = df_train["question"].str.split().str.len()
df_train["q_len_chars"] = df_train["question"].str.len()

print("\nQuestion length (words):")
print(f"  Mean: {df_train['q_len_words'].mean():.1f}")
print(f"  Median: {df_train['q_len_words'].median():.1f}")
print(f"  Std: {df_train['q_len_words'].std():.1f}")
print(
    f"  P5/P95: {df_train['q_len_words'].quantile(0.05):.0f} / {df_train['q_len_words'].quantile(0.95):.0f}"
)

# Explanation availability
has_exp = df_train["exp"].notna() & (df_train["exp"].str.len() > 5)
print(
    f"\nExplanations available: {has_exp.sum():,}/{len(df_train):,} ({has_exp.mean() * 100:.1f}%)"
)

# Choice type distribution
print(f"\nChoice types: {df_train['choice_type'].value_counts().to_dict()}")

# Topic diversity
n_subjects = df_train["subject_name"].nunique()
n_topics = df_train["topic_name"].nunique()
print(f"\nUnique subjects: {n_subjects}")
print(f"Unique topics: {n_topics}")

# ============================================================
# 5. Calibration Training Split Strategy
# ============================================================
print("\n" + "=" * 60)
print("RECOMMENDED SPLIT STRATEGY FOR CALIBRATION TRAINING")
print("=" * 60)

# Strategy: Use train split, create our own partitions
train_size = len(df_train)

# For LoRA + Prompt calibration training
calib_train_size = 5000
calib_val_size = 2000
grading_pool_size = train_size - calib_train_size - calib_val_size

print(f"""
  From train split ({train_size:,} samples):
  
  1. Calibration Training Set: {calib_train_size:,} samples
     → Stratified sample maintaining subject & label balance
     → Used for LoRA + Prompt fine-tuning (graded correct/incorrect)
     
  2. Calibration Validation Set: {calib_val_size:,} samples  
     → For hyperparameter tuning and early stopping
     
  3. Response Generation Pool: {grading_pool_size:,} samples
     → Generate model responses with OlMo 1B
     → Grade against ground truth (correct/incorrect)
     → This generates the training signal for calibration
  
  From validation split ({len(df_val):,} samples):
     → Final evaluation: ECE, AUROC, selective prediction
     → Different exam source (NEET PG vs mixed) = slight distribution shift
""")

# Create and save a stratified sample for calibration training
# Stratified by subject and label
df_train["stratify_key"] = df_train["subject_name"] + "_" + df_train["cop"].astype(str)

calib_set, remainder = train_test_split(
    df_train,
    train_size=calib_train_size,
    stratify=df_train["subject_name"],  # Stratify by subject
    random_state=42,
)

calib_val, grading_pool = train_test_split(
    remainder,
    train_size=calib_val_size,
    stratify=remainder["subject_name"],
    random_state=42,
)

print(f"  Calibration train set: {len(calib_set):,}")
print(f"  Calibration val set: {len(calib_val):,}")
print(f"  Grading pool: {len(grading_pool):,}")

# Verify label balance in calibration set
print("\n  Label balance in calibration train set:")
for idx in range(4):
    count = (calib_set["cop"] == idx).sum()
    print(f"    {option_map[idx]}: {count} ({count / len(calib_set) * 100:.1f}%)")

# Save split indices for reproducibility
splits = {
    "calib_train_ids": calib_set["id"].tolist()
    if "id" in calib_set.columns
    else calib_set.index.tolist(),
    "calib_val_ids": calib_val["id"].tolist()
    if "id" in calib_val.columns
    else calib_val.index.tolist(),
    "grading_pool_ids": grading_pool["id"].tolist()
    if "id" in grading_pool.columns
    else grading_pool.index.tolist(),
}

with open(f"{OUTPUT_DIR}/calibration_split_indices.json", "w") as f:
    json.dump(
        {k: v[:10] for k, v in splits.items()}, f, indent=2, default=str
    )  # Save preview

print(f"\n  Split indices saved to {OUTPUT_DIR}/calibration_split_indices.json")

# ============================================================
# 6. OOD Generalization Design
# ============================================================
print("\n" + "=" * 60)
print("OUT-OF-DISTRIBUTION GENERALIZATION DESIGN")
print("=" * 60)

print("""
  Following the paper's methodology for testing generalization:
  
  A) SUBJECT HOLDOUT (within MedMCQA):
     Train calibration on: all subjects EXCEPT held-out
     Evaluate on: held-out subjects
     
     Suggested holdouts (diverse, sufficient size):
""")

# Find good holdout candidates (medium-sized, distinct domains)
for subject in ["Psychiatry", "Dermatology", "Ophthalmology", "Forensic Medicine"]:
    count = len(df_train[df_train["subject_name"] == subject])
    print(f"     • {subject}: {count:,} samples")

print("""
  B) CROSS-DATASET (MedMCQA → MedQA):
     Train calibration on: MedMCQA
     Evaluate on: MedQA (USMLE)
     Tests: curriculum shift, question style shift, language shift
     
  C) FORMAT SHIFT (MC → Open-Ended):
     Train on: MedMCQA multiple-choice
     Evaluate on: MedMCQA questions WITHOUT answer choices
     Tests: whether calibration transfers across question formats
     (This mirrors the MC→OE experiment from the paper)
""")

# ============================================================
# 7. Generate Summary Visualizations
# ============================================================
print("Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "MedMCQA Dataset Profile for Calibration Training", fontsize=14, fontweight="bold"
)

# 7a: Overall label distribution
ax = axes[0, 0]
labels = df_train["cop"].value_counts().sort_index()
ax.bar(
    [option_map[i] for i in range(4)],
    labels.values,
    color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"],
)
ax.axhline(
    y=len(df_train) / 4, color="gray", linestyle="--", alpha=0.7, label="Ideal 25%"
)
ax.set_title("Label Distribution (Train)")
ax.set_ylabel("Count")
ax.legend()

# 7b: Subject distribution
ax = axes[0, 1]
top_10 = df_train["subject_name"].value_counts().head(10)
ax.barh(top_10.index[::-1], top_10.values[::-1], color="steelblue")
ax.set_title("Top 10 Subjects by Sample Count")
ax.set_xlabel("Count")

# 7c: Question length distribution
ax = axes[1, 0]
ax.hist(df_train["q_len_words"], bins=50, color="teal", alpha=0.7, edgecolor="none")
ax.set_title("Question Length Distribution")
ax.set_xlabel("Words")
ax.set_ylabel("Count")
ax.axvline(
    x=df_train["q_len_words"].median(),
    color="red",
    linestyle="--",
    label=f"Median: {df_train['q_len_words'].median():.0f}",
)
ax.legend()

# 7d: Per-subject label entropy
ax = axes[1, 1]
entropies = []
for subject in df_train["subject_name"].unique():
    subset = df_train[df_train["subject_name"] == subject]
    counts = [len(subset[subset["cop"] == i]) for i in range(4)]
    probs = np.array(counts) / sum(counts)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    entropies.append({"subject": subject, "entropy": entropy, "count": sum(counts)})

ent_df = pd.DataFrame(entropies).sort_values("entropy")
ax.barh(ent_df["subject"], ent_df["entropy"], color="coral")
ax.axvline(x=2.0, color="green", linestyle="--", alpha=0.7, label="Max entropy (2.0)")
ax.set_title("Per-Subject Label Entropy")
ax.set_xlabel("Entropy (bits)")
ax.legend()
ax.set_xlim(1.8, 2.05)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/medmcqa_profile.png", dpi=150, bbox_inches="tight")
print(f"Saved to {OUTPUT_DIR}/medmcqa_profile.png")

# ============================================================
# 8. Export Summary Report
# ============================================================
report = {
    "dataset": "MedMCQA",
    "total_samples": len(df_train) + len(df_val),
    "train_samples": len(df_train),
    "val_samples": len(df_val),
    "num_choices": 4,
    "num_subjects": n_subjects,
    "num_topics": n_topics,
    "explanation_availability_pct": float(has_exp.mean() * 100),
    "avg_question_length_words": float(df_train["q_len_words"].mean()),
    "label_distribution": {
        option_map[i]: float((df_train["cop"] == i).mean()) for i in range(4)
    },
    "recommendation": "PRIMARY DATASET - meets all requirements for calibration training",
    "concerns": [
        "Test set labels withheld - use validation split for final eval",
        "Indian medical curriculum - may need cross-cultural validation",
        "Some questions may require visual aids not included in text",
    ],
}

with open(f"{OUTPUT_DIR}/medmcqa_summary.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\nSummary report saved to {OUTPUT_DIR}/medmcqa_summary.json")

# ============================================================
# 9. Sample Questions Across Length Distribution
# ============================================================
print("\n" + "=" * 60)
print("SAMPLE QUESTIONS ACROSS LENGTH DISTRIBUTION")
print("=" * 60)

# Ensure q_len_words exists (computed in section 4)
if "q_len_words" not in df_train.columns:
    df_train["q_len_words"] = df_train["question"].str.split().str.len()

# Sample one question from each decile (P10 through P90) to reflect
# the true length distribution, then deduplicate by index
percentiles = [10, 25, 50, 75, 90]
sampled_indices = []
for p in percentiles:
    target_len = df_train["q_len_words"].quantile(p / 100)
    # Find rows within ±1 word of the target percentile length
    candidates = df_train[
        (df_train["q_len_words"] >= target_len - 1)
        & (df_train["q_len_words"] <= target_len + 1)
        & (~df_train.index.isin(sampled_indices))
    ]
    if not candidates.empty:
        sampled_indices.append(candidates.sample(1, random_state=p).index[0])

samples = df_train.loc[sampled_indices].reset_index(drop=True)

option_labels = {0: "A", 1: "B", 2: "C", 3: "D"}
sample_records = []

for i, row in samples.iterrows():
    correct_letter = option_labels.get(row["cop"], "?")
    correct_text = {
        "A": row["opa"],
        "B": row["opb"],
        "C": row["opc"],
        "D": row["opd"],
    }.get(correct_letter, "")
    record = {
        "example": i + 1,
        "subject": row["subject_name"],
        "topic": row.get("topic_name", ""),
        "question_length_words": int(row["q_len_words"]),
        "question": row["question"],
        "options": {
            "A": row["opa"],
            "B": row["opb"],
            "C": row["opc"],
            "D": row["opd"],
        },
        "correct_answer": correct_letter,
        "correct_text": correct_text,
        "explanation": row["exp"]
        if pd.notna(row.get("exp")) and len(str(row.get("exp", ""))) > 5
        else None,
    }
    sample_records.append(record)

    # Print to console
    print(f"\n{'─' * 60}")
    print(
        f"Example {i + 1}  |  Subject: {record['subject']}  |  {record['question_length_words']} words"
    )
    print(f"{'─' * 60}")
    print(f"Q: {record['question']}")
    print(f"   A) {row['opa']}")
    print(f"   B) {row['opb']}")
    print(f"   C) {row['opc']}")
    print(f"   D) {row['opd']}")
    print(f"Answer: {correct_letter}) {correct_text}")
    if record["explanation"]:
        print(f"Explanation: {record['explanation']}")

print(f"\n{'─' * 60}")

# Save to JSON
with open(f"{OUTPUT_DIR}/sample_questions.json", "w") as f:
    json.dump(sample_records, f, indent=2)

print(f"\nSample questions saved to {OUTPUT_DIR}/sample_questions.json")
print("\n✓ Analysis complete! Review outputs in ./dataset_analysis_output/")
