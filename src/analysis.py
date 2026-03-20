"""
Natural Hallucinations: Analysis and Visualization.

Reads experiment results from results/raw/ and produces:
- Statistical analysis (results/analysis.json)
- Visualizations (results/plots/)
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path("/workspaces/natural-hallucinations-claude")
RAW_DIR = BASE_DIR / "results" / "raw"
PLOT_DIR = BASE_DIR / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
MODEL_LABELS = {"gpt-4.1": "GPT-4.1", "gpt-4o": "GPT-4o",
                "gpt-4o-mini": "GPT-4o-mini", "gpt-3.5-turbo": "GPT-3.5"}

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.bbox': 'tight'})


def load_results():
    """Load all experiment results."""
    results = {}
    for name in ["exp1_cross_model", "exp2_robustness", "exp3_self_detection", "exp4_transfer"]:
        path = RAW_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


def analyze_exp1(exp1):
    """Analyze cross-model hallucination rates."""
    n = len(exp1[MODELS[0]])

    # Per-model rates
    rates = {}
    for m in MODELS:
        truthful = sum(1 for r in exp1[m] if r["truthful"])
        rates[m] = {"truthful": truthful, "hallucinated": n - truthful,
                     "rate": round((n - truthful) / n, 4)}

    # Per-question hallucination frequency
    freq = np.zeros(n)
    for m in MODELS:
        for r in exp1[m]:
            if not r["truthful"]:
                freq[r["idx"]] += 1

    freq_dist = {int(k): int(v) for k, v in zip(*np.unique(freq, return_counts=True))}

    # Category analysis
    cats = {}
    for r in exp1[MODELS[0]]:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(r["idx"])

    cat_rates = {}
    for cat, idxs in sorted(cats.items()):
        cf = freq[idxs]
        cat_rates[cat] = {
            "n": len(idxs),
            "mean_models_hallu": round(float(np.mean(cf)), 2),
            "all_models_hallu": int(np.sum(cf == len(MODELS))),
            "any_model_hallu": int(np.sum(cf > 0)),
        }

    return {"per_model": rates, "freq_distribution": freq_dist,
            "category_rates": cat_rates, "n_questions": n}


def analyze_exp2(exp2):
    """Analyze robustness results."""
    all_persistence = []
    per_model = {}

    for m in MODELS:
        if m not in exp2:
            continue
        hallu_only = [r for r in exp2[m] if not r["original_truthful"]]
        if hallu_only:
            rates = [r["persistence_rate"] for r in hallu_only]
            per_model[m] = {
                "n": len(hallu_only),
                "mean_persistence": round(float(np.mean(rates)), 4),
                "std_persistence": round(float(np.std(rates)), 4),
                "median_persistence": round(float(np.median(rates)), 4),
                "fully_persistent": int(sum(1 for r in rates if r == 1.0)),
            }
            all_persistence.extend(rates)

    overall = {
        "mean": round(float(np.mean(all_persistence)), 4) if all_persistence else 0,
        "std": round(float(np.std(all_persistence)), 4) if all_persistence else 0,
        "n": len(all_persistence),
    }

    return {"per_model": per_model, "overall": overall}


def analyze_exp3(exp3):
    """Analyze self-detection results."""
    per_model = {}
    all_hallu_acc = []
    all_ctrl_acc = []

    for m in MODELS:
        if m not in exp3:
            continue
        h = exp3[m]["hallucinated"]
        t = exp3[m]["truthful_control"]
        ha = [r["detected"] for r in h]
        ta = [r["detected"] for r in t]

        per_model[m] = {
            "hallu_accuracy": round(float(np.mean(ha)), 4) if ha else 0,
            "control_accuracy": round(float(np.mean(ta)), 4) if ta else 0,
            "hallu_n": len(ha),
            "control_n": len(ta),
            "gap": round(float(np.mean(ta) - np.mean(ha)), 4) if ha and ta else 0,
        }
        all_hallu_acc.extend(ha)
        all_ctrl_acc.extend(ta)

    # Statistical test: are hallucination detection rates significantly lower?
    if all_hallu_acc and all_ctrl_acc:
        # Use a two-proportion z-test
        p1 = np.mean(all_hallu_acc)
        p2 = np.mean(all_ctrl_acc)
        n1, n2 = len(all_hallu_acc), len(all_ctrl_acc)
        p_pool = (sum(all_hallu_acc) + sum(all_ctrl_acc)) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z = (p2 - p1) / se if se > 0 else 0
        p_val = 1 - stats.norm.cdf(z)  # one-tailed
        test = {"z_stat": round(z, 4), "p_value": round(p_val, 6), "effect_size": round(p2 - p1, 4)}
    else:
        test = {}

    return {"per_model": per_model, "statistical_test": test}


def analyze_exp4(exp4):
    """Analyze transfer and temporal results (already computed, just format)."""
    return exp4  # Already well-structured


# ── Visualization ──

def plot_exp1_halluc_rates(exp1_analysis):
    """Bar chart of per-model hallucination rates."""
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(exp1_analysis["per_model"].keys())
    rates = [exp1_analysis["per_model"][m]["rate"] * 100 for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    bars = ax.bar(labels, rates, color=sns.color_palette("Set2", len(models)), edgecolor='black', linewidth=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Hallucination Rate (%)')
    ax.set_title('TruthfulQA Hallucination Rates by Model')
    ax.set_ylim(0, max(rates) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(PLOT_DIR / 'exp1_halluc_rates.png')
    plt.close()


def plot_exp1_freq_dist(exp1_analysis):
    """Histogram of per-question hallucination frequency."""
    fig, ax = plt.subplots(figsize=(7, 5))
    freq_dist = exp1_analysis["freq_distribution"]
    x = sorted([int(k) for k in freq_dist.keys()])
    y = [freq_dist[str(k)] if str(k) in freq_dist else freq_dist.get(k, 0) for k in x]

    colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b'][:len(x)]
    bars = ax.bar(x, y, color=colors, edgecolor='black', linewidth=0.5)
    for bar, count in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(count), ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Number of Models Hallucinating')
    ax.set_ylabel('Number of Questions')
    ax.set_title('Distribution of Cross-Model Hallucination Frequency')
    ax.set_xticks(x)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(PLOT_DIR / 'exp1_freq_distribution.png')
    plt.close()


def plot_exp1_category(exp1_analysis):
    """Horizontal bar chart of category-level hallucination rates."""
    cats = exp1_analysis["category_rates"]
    # Sort by mean hallucination
    sorted_cats = sorted(cats.items(), key=lambda x: x[1]["mean_models_hallu"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_cats) * 0.35)))
    names = [c[0] for c in sorted_cats]
    means = [c[1]["mean_models_hallu"] for c in sorted_cats]

    colors = plt.cm.RdYlGn_r(np.array(means) / max(means) if max(means) > 0 else np.zeros(len(means)))
    ax.barh(names, means, color=colors, edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Mean Number of Models Hallucinating')
    ax.set_title('Hallucination by TruthfulQA Category')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(PLOT_DIR / 'exp1_category_rates.png')
    plt.close()


def plot_exp2_persistence(exp2_analysis):
    """Box plot of persistence rates per model."""
    # Need raw data for box plot - reload
    with open(RAW_DIR / "exp2_robustness.json") as f:
        exp2 = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 5))
    data, labels = [], []
    for m in MODELS:
        if m not in exp2:
            continue
        rates = [r["persistence_rate"] for r in exp2[m] if not r["original_truthful"]]
        if rates:
            data.append(rates)
            labels.append(MODEL_LABELS.get(m, m))

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = sns.color_palette("Set2", len(data))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_ylabel('Persistence Rate')
        ax.set_title('Hallucination Persistence Under Rephrasing')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.savefig(PLOT_DIR / 'exp2_persistence.png')
    plt.close()


def plot_exp3_detection(exp3_analysis):
    """Grouped bar chart of self-detection accuracy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    models = [m for m in MODELS if m in exp3_analysis["per_model"]]
    labels = [MODEL_LABELS.get(m, m) for m in models]
    hallu_acc = [exp3_analysis["per_model"][m]["hallu_accuracy"] * 100 for m in models]
    ctrl_acc = [exp3_analysis["per_model"][m]["control_accuracy"] * 100 for m in models]

    x = np.arange(len(models))
    w = 0.35
    bars1 = ax.bar(x - w/2, hallu_acc, w, label='Own Hallucinations', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + w/2, ctrl_acc, w, label='Control (Truthful)', color='#2ecc71', alpha=0.8)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}%',
                    ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Detection Accuracy (%)')
    ax.set_title('Self-Detection: Hallucinated vs Control Questions')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Random')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(PLOT_DIR / 'exp3_self_detection.png')
    plt.close()


def plot_exp4_jaccard(exp4):
    """Heatmap of pairwise Jaccard similarity."""
    fig, ax = plt.subplots(figsize=(7, 6))
    n = len(MODELS)
    matrix = np.zeros((n, n))

    for i, m1 in enumerate(MODELS):
        matrix[i, i] = 1.0
        for j, m2 in enumerate(MODELS):
            if i < j:
                pair = f"{m1} vs {m2}"
                if pair in exp4["jaccard"]:
                    jac = exp4["jaccard"][pair]["jaccard"]
                    matrix[i, j] = jac
                    matrix[j, i] = jac

    labels = [MODEL_LABELS.get(m, m) for m in MODELS]
    sns.heatmap(matrix, annot=True, fmt='.3f', xticklabels=labels, yticklabels=labels,
                cmap='YlOrRd', ax=ax, vmin=0, vmax=1, linewidths=0.5)
    ax.set_title('Cross-Model Jaccard Similarity of Hallucinated Questions')
    plt.savefig(PLOT_DIR / 'exp4_jaccard_heatmap.png')
    plt.close()


def plot_exp4_temporal(exp4):
    """Contingency table visualization for temporal prediction."""
    if "temporal" not in exp4:
        return
    t = exp4["temporal"]
    ct = np.array(t["contingency"])

    fig, ax = plt.subplots(figsize=(6, 5))
    labels_x = ['Truthful', 'Hallucinated']
    labels_y = ['Truthful', 'Hallucinated']

    sns.heatmap(ct, annot=True, fmt='d', xticklabels=labels_x, yticklabels=labels_y,
                cmap='Blues', ax=ax, linewidths=0.5)
    ax.set_xlabel(f'{MODEL_LABELS.get(t["new_model"], t["new_model"])}')
    ax.set_ylabel(f'{MODEL_LABELS.get(t["old_model"], t["old_model"])}')
    ax.set_title(f'Temporal Prediction: r={t["pearson_r"]:.3f} (p={t["pearson_p"]:.4f})')
    plt.savefig(PLOT_DIR / 'exp4_temporal_contingency.png')
    plt.close()


def main():
    print("Loading results...")
    results = load_results()

    analysis = {}

    if "exp1_cross_model" in results:
        print("Analyzing Experiment 1...")
        a1 = analyze_exp1(results["exp1_cross_model"])
        analysis["exp1"] = a1
        plot_exp1_halluc_rates(a1)
        plot_exp1_freq_dist(a1)
        plot_exp1_category(a1)
        for m in MODELS:
            rate = a1["per_model"][m]["rate"]*100
            print(f"  {MODEL_LABELS[m]}: {rate:.1f}%")
        print(f"  Freq dist: {a1['freq_distribution']}")

    if "exp2_robustness" in results:
        print("Analyzing Experiment 2...")
        a2 = analyze_exp2(results["exp2_robustness"])
        analysis["exp2"] = a2
        plot_exp2_persistence(a2)
        print(f"  Overall persistence: {a2['overall']['mean']:.2%} ± {a2['overall']['std']:.2%}")

    if "exp3_self_detection" in results:
        print("Analyzing Experiment 3...")
        a3 = analyze_exp3(results["exp3_self_detection"])
        analysis["exp3"] = a3
        plot_exp3_detection(a3)
        for m in MODELS:
            if m in a3["per_model"]:
                d = a3["per_model"][m]
                print(f"  {MODEL_LABELS[m]}: hallu={d['hallu_accuracy']:.2%}, ctrl={d['control_accuracy']:.2%}, gap={d['gap']:.2%}")

    if "exp4_transfer" in results:
        print("Analyzing Experiment 4...")
        a4 = analyze_exp4(results["exp4_transfer"])
        analysis["exp4"] = a4
        plot_exp4_jaccard(a4)
        plot_exp4_temporal(a4)

    # Save analysis
    with open(BASE_DIR / "results" / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    print("\nAnalysis complete! Plots saved to results/plots/")
    return analysis


if __name__ == "__main__":
    main()
