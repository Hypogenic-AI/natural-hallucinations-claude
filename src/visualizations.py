"""
Create visualizations for Natural Hallucinations experiments.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

RESULTS_DIR = Path("/data/hypogenicai/workspaces/natural-hallucinations-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_results():
    """Load all experiment results."""
    results = {}

    for fname in ["all_results_summary.json", "exp1_results.json", "exp2_results.json",
                  "exp3_results.json", "exp4_results.json"]:
        path = RESULTS_DIR / fname
        if path.exists():
            with open(path) as f:
                results[fname.replace(".json", "")] = json.load(f)

    return results


def plot_model_accuracy(results):
    """Plot accuracy comparison across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    exp1 = results["all_results_summary"]["experiment_1"]
    models = list(exp1["per_model_accuracy"].keys())
    accuracies = [exp1["per_model_accuracy"][m] * 100 for m in models]

    # Clean model names
    model_labels = ["GPT-4o", "GPT-3.5-turbo", "Claude 3.5 Sonnet", "Llama 3 70B"]

    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
    bars = ax.bar(model_labels, accuracies, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.0f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Truthfulness Rate (%)', fontsize=12)
    ax.set_title('Model Accuracy on TruthfulQA (100 Questions)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random baseline')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: model_accuracy.png")


def plot_error_correlation_matrix(results):
    """Plot error correlation matrix between models."""
    exp1 = results["exp1_results"]

    # Build error matrix
    models = list(exp1["model_correct"].keys())
    n_questions = len(exp1["model_correct"][models[0]])

    # Create DataFrame of errors (0=error, 1=correct)
    error_df = pd.DataFrame({
        model: [1 if c else 0 for c in exp1["model_correct"][model]]
        for model in models
    })

    # Calculate correlation
    corr_matrix = error_df.corr()

    # Clean model names
    clean_names = ["GPT-4o", "GPT-3.5", "Claude 3.5", "Llama 3"]
    corr_matrix.index = clean_names
    corr_matrix.columns = clean_names

    fig, ax = plt.subplots(figsize=(8, 7))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0,
                square=True, linewidths=2, fmt='.2f',
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                ax=ax, annot_kws={"size": 14})

    ax.set_title('Cross-Model Error Correlation\n(Higher = Similar Error Patterns)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "error_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: error_correlation.png")


def plot_natural_hallucination_categories(results):
    """Plot categories of natural hallucinations."""
    exp1 = results["exp1_results"]

    natural_halls = [q for q in exp1["per_question_results"] if q["is_natural_hallucination"]]

    categories = [q["category"] for q in natural_halls]
    category_counts = Counter(categories)

    fig, ax = plt.subplots(figsize=(10, 6))

    cats = list(category_counts.keys())
    counts = list(category_counts.values())

    colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))
    bars = ax.barh(cats, counts, color=colors, edgecolor='black', linewidth=1.2)

    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax.annotate(f'{count}',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(3, 0),
                   textcoords="offset points",
                   ha='left', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Number of Natural Hallucinations', fontsize=12)
    ax.set_title('Categories of Natural Hallucinations\n(Questions Where 3+ Models Failed)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(counts) + 1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "natural_hall_categories.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: natural_hall_categories.png")


def plot_robustness_scores(results):
    """Plot robustness of natural hallucinations to paraphrasing."""
    exp2 = results.get("exp2_results", {})

    if not exp2 or "questions" not in exp2:
        print("No exp2 data for robustness plot")
        return

    robustness_scores = [q["robustness_score"] for q in exp2["questions"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    bins = [0, 0.33, 0.67, 1.01]
    labels = ['0/3 paraphrases\nstill fail', '1/3 paraphrases\nstill fail',
              '2-3/3 paraphrases\nstill fail']
    counts = [0, 0, 0]
    for score in robustness_scores:
        if score < 0.34:
            counts[0] += 1
        elif score < 0.67:
            counts[1] += 1
        else:
            counts[2] += 1

    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    bars = ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=1.5)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{count}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Number of Questions', fontsize=12)
    ax.set_title(f'Robustness of Natural Hallucinations to Paraphrasing\n(n={len(robustness_scores)} questions, Mean robustness: {np.mean(robustness_scores):.1%})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(counts) + 1.5)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "robustness_paraphrase.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: robustness_paraphrase.png")


def plot_self_recognition(results):
    """Plot self-recognition comparison."""
    exp3 = results.get("exp3_results", {})

    if not exp3 or "summary" not in exp3:
        # Use all_results_summary
        exp3 = {"summary": results["all_results_summary"]["experiment_3"]}

    summary = exp3["summary"]

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['Natural\nHallucinations', 'Model-Specific\nErrors']
    rates = [summary["natural_hallucination_recognition_rate"] * 100,
             summary["model_specific_recognition_rate"] * 100]

    colors = ['#e74c3c', '#3498db']
    bars = ax.bar(categories, rates, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.annotate(f'{rate:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Self-Recognition Rate (%)', fontsize=12)
    ax.set_title('Can Models Recognize Their Own Errors?\n(Recognition = Model says its wrong answer is wrong)',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 50)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.text(1.5, 51, 'Random guessing', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "self_recognition.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: self_recognition.png")


def plot_temporal_analysis(results):
    """Plot temporal analysis: GPT-3.5 vs GPT-4o."""
    exp4 = results["all_results_summary"]["experiment_4"]
    ct = exp4["contingency_table"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Contingency heatmap
    data = np.array([[ct["both_correct"], ct["gpt35_only_wrong"]],
                     [ct["gpt4o_only_wrong"], ct["both_wrong"]]])

    im = ax1.imshow(data, cmap='YlOrRd')

    # Add labels
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['GPT-3.5\nCorrect', 'GPT-3.5\nWrong'])
    ax1.set_yticklabels(['GPT-4o\nCorrect', 'GPT-4o\nWrong'])

    # Add values
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, data[i, j],
                           ha="center", va="center", color="black",
                           fontsize=16, fontweight='bold')

    ax1.set_title('Error Contingency Table\n(GPT-3.5 vs GPT-4o)', fontsize=14, fontweight='bold')

    # Right: Bar chart comparing
    categories = ['Both Correct', 'Only GPT-3.5\nWrong', 'Only GPT-4o\nWrong', 'Both Wrong']
    values = [ct["both_correct"], ct["gpt35_only_wrong"],
              ct["gpt4o_only_wrong"], ct["both_wrong"]]
    colors = ['#2ecc71', '#f1c40f', '#9b59b6', '#e74c3c']

    bars = ax2.bar(categories, values, color=colors, edgecolor='black', linewidth=1.2)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.annotate(f'{val}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylabel('Number of Questions', fontsize=12)
    ax2.set_title(f'Temporal Error Analysis\n(Pearson r = {exp4["correlation"]["pearson_r"]:.3f}, p < 0.001)',
                 fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(values) + 10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "temporal_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: temporal_analysis.png")


def plot_n_models_wrong_distribution(results):
    """Plot distribution of how many models get each question wrong."""
    exp1 = results["exp1_results"]

    n_wrong_counts = [q["n_models_incorrect"] for q in exp1["per_question_results"]]
    counts = Counter(n_wrong_counts)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = list(range(5))
    y = [counts.get(i, 0) for i in x]
    colors = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']

    bars = ax.bar(x, y, color=colors, edgecolor='black', linewidth=1.5)

    for bar, count in zip(bars, y):
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{count}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Number of Models Getting Question Wrong', fontsize=12)
    ax.set_ylabel('Number of Questions', fontsize=12)
    ax.set_title('Distribution of Error Agreement Across Models\n(4 models tested)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['0\n(All correct)', '1\n(Model-specific)', '2\n(Partial)', '3\n(Natural hall.)', '4\n(Universal hall.)'])

    # Add region annotations
    ax.axvspan(2.5, 4.5, alpha=0.1, color='red')
    ax.text(3.5, max(y) * 0.9, '"Natural\nHallucinations"', ha='center', fontsize=11, color='darkred')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "error_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: error_distribution.png")


def create_summary_table(results):
    """Create a summary table of all findings."""
    summary = results["all_results_summary"]

    print("\n" + "="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)

    print("\nExperiment 1: Cross-Model Transfer")
    print("-" * 40)
    print(f"  Questions tested: {summary['experiment_1']['n_questions']}")
    for model, acc in summary['experiment_1']['per_model_accuracy'].items():
        print(f"  {model}: {acc:.1%}")
    print(f"  Natural hallucinations found: {summary['experiment_1']['natural_hallucination_count']}")

    print("\nExperiment 2: Robustness to Paraphrasing")
    print("-" * 40)
    print(f"  Questions tested: {summary['experiment_2']['n_questions_tested']}")
    print(f"  Mean robustness score: {summary['experiment_2']['mean_robustness_score']:.1%}")
    print(f"  Highly robust (2/3+ paraphrases): {summary['experiment_2']['n_highly_robust']}")

    print("\nExperiment 3: Self-Recognition")
    print("-" * 40)
    print(f"  Natural hallucination recognition: {summary['experiment_3']['natural_hallucination_recognition_rate']:.1%}")
    print(f"  Model-specific error recognition: {summary['experiment_3']['model_specific_recognition_rate']:.1%}")
    print(f"  Recognition gap: {summary['experiment_3']['recognition_gap']:.1%}")

    print("\nExperiment 4: Temporal Analysis (GPT-3.5 → GPT-4o)")
    print("-" * 40)
    print(f"  GPT-3.5 accuracy: {summary['experiment_4']['summary']['gpt35_accuracy']:.1%}")
    print(f"  GPT-4o accuracy: {summary['experiment_4']['summary']['gpt4o_accuracy']:.1%}")
    print(f"  Pearson correlation: {summary['experiment_4']['correlation']['pearson_r']:.3f}")
    print(f"  P(GPT-4o wrong | GPT-3.5 wrong): {summary['experiment_4']['predictive_power']['p_gpt4o_wrong_given_gpt35_wrong']:.1%}")


def main():
    """Generate all visualizations."""
    print("Loading results...")
    results = load_results()

    print("\nGenerating visualizations...")
    plot_model_accuracy(results)
    plot_error_correlation_matrix(results)
    plot_natural_hallucination_categories(results)
    plot_robustness_scores(results)
    plot_self_recognition(results)
    plot_temporal_analysis(results)
    plot_n_models_wrong_distribution(results)

    create_summary_table(results)

    print(f"\nAll visualizations saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
