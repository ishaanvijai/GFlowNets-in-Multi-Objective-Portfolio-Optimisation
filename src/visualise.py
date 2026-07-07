import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — must come before pyplot import
import matplotlib.pyplot as plt

EVAL_DIR = "evals"


def _save(name):
    """Save the current figure to EVAL_DIR/<name> and close it."""
    os.makedirs(EVAL_DIR, exist_ok=True)
    plt.savefig(os.path.join(EVAL_DIR, name), bbox_inches="tight", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Baseline (random-only) plots
# ---------------------------------------------------------------------------

def plot_pareto_scatter(df):
    """Pareto scatter for a single distribution (used for the random baseline)."""
    plt.scatter(df["sharpe"], df["mdd"], c=np.asarray(df["rank"]), cmap="magma", s=15, alpha=0.7)
    plt.colorbar(label="Pareto rank")
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Max Drawdown")
    plt.title("Random baseline — Pareto scatter (train)")
    plt.grid()
    _save("01_pareto_scatter.png")


def plot_combined_reward(df):
    plt.plot(df["reward"])
    plt.xlabel("Sample")
    plt.ylabel("Pareto-based Reward")
    plt.title("Pareto-based Reward per Sample")
    plt.grid()
    _save("02_combined_reward.png")


def plot_rank_histogram(df):
    plt.hist(df["rank"], bins=50, edgecolor='black', alpha=0.7)
    plt.grid()
    plt.xlabel("Pareto Rank")
    plt.ylabel("Frequency")
    _save("03_rank_histogram.png")


def plot_sharpe_histogram(df):
    plt.hist(df["sharpe"], bins=50, edgecolor='black', alpha=0.7)
    plt.grid()
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Frequency")
    _save("04_sharpe_histogram.png")


def plot_mdd_histogram(df):
    plt.hist(df["mdd"], bins=50, edgecolor='black', alpha=0.7)
    plt.grid()
    plt.xlabel("Max Drawdown")
    plt.ylabel("Frequency")
    _save("05_mdd_histogram.png")


# ---------------------------------------------------------------------------
# Three-way comparison scatter (Random cloud / GFN cloud / EW point)
# ---------------------------------------------------------------------------

def plot_pareto_scatter_comparison(rand_df, gfn_df, ew_sharpe, ew_mdd,
                                   title="Portfolio comparison: Sharpe vs MDD",
                                   save_name="pareto_scatter_comparison.png"):
    """Scatter of Sharpe vs MDD for all three strategies.

    Random and GFlowNet are shown as clouds; Equal-Weight as a single star.
    Lower MDD and higher Sharpe = top-left corner = better.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(rand_df["sharpe"], rand_df["mdd"],
                color="lightblue", alpha=0.4, s=12, label="Random")
    plt.scatter(gfn_df["sharpe"], gfn_df["mdd"],
                color="darkblue", alpha=0.5, s=12, label="GFlowNet")
    plt.scatter([ew_sharpe], [ew_mdd],
                color="orange", marker="*", s=400, zorder=5,
                edgecolors="black", linewidths=0.5, label="Equal-Weight (1/N)")
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Max Drawdown")
    plt.title(title)
    plt.legend()
    plt.grid()
    _save(save_name)


# ---------------------------------------------------------------------------
# Walk-forward CV results (3 bars per fold)
# ---------------------------------------------------------------------------

def plot_walk_forward_results(rand_rewards, gfn_rewards, ew_rewards):
    """Grouped bar chart: per-fold dev Pareto reward for all three strategies."""
    folds = np.arange(1, len(rand_rewards) + 1)
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(folds - width, rand_rewards, width, label="Random", color="lightblue")
    ax.bar(folds,          gfn_rewards,  width, label="GFlowNet", color="darkblue")
    ax.bar(folds + width,  ew_rewards,   width, label="Equal-Weight", color="orange")

    ax.axhline(np.mean(rand_rewards), color="steelblue",  linestyle="--", linewidth=1.2, label="Random mean")
    ax.axhline(np.mean(gfn_rewards),  color="navy",       linestyle="--", linewidth=1.2, label="GFlowNet mean")
    ax.axhline(np.mean(ew_rewards),   color="darkorange", linestyle="--", linewidth=1.2, label="EW mean")

    ax.set_xlabel("Fold")
    ax.set_ylabel("Dev reward (Pareto-based)")
    ax.set_title("Walk-forward cross-validation — dev reward per fold")
    ax.set_xticks(folds)
    ax.legend(fontsize=8)
    ax.grid(axis="y")
    plt.tight_layout()
    _save("06_walk_forward_results.png")


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(lossarray):
    plt.plot(lossarray)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid()
    _save("07_loss_curve.png")


# ---------------------------------------------------------------------------
# 50-trial repeated experiment plots (all three strategies)
# ---------------------------------------------------------------------------

def plot_trial_comparison(rand_rewards, eval_rewards, ew_rewards):
    """Scatter of per-trial mean rewards with trend lines for all three methods."""
    nx = np.arange(len(rand_rewards))

    plt.scatter(nx, rand_rewards, color="lightblue", label="Random", alpha=0.5, s=20)
    plt.scatter(nx, eval_rewards, color="darkblue",  label="GFlowNet", alpha=0.7, s=20)
    plt.scatter(nx, ew_rewards,   color="orange",    label="Equal-Weight", alpha=0.7, s=20, marker="D")
    plt.grid()
    plt.ylabel("Pareto-based Reward")
    plt.xlabel("Trial")
    plt.title("Random vs GFlowNet vs Equal-Weight (higher is better)")

    for vals, colour, lbl in [
        (rand_rewards, "steelblue",  "Random Trend"),
        (eval_rewards, "navy",       "GFlowNet Trend"),
        (ew_rewards,   "darkorange", "EW Trend"),
    ]:
        z = np.polyfit(nx, vals, 1)
        plt.plot(nx, np.poly1d(z)(nx), color=colour, linestyle="solid", linewidth=2.5, label=lbl)

    plt.legend(fontsize=8)
    _save("08_trial_comparison.png")


def plot_reward_boxplot(rand_rewards, eval_rewards, ew_rewards):
    """Box plot comparing Pareto reward distribution across all three methods."""
    plt.boxplot(
        [rand_rewards, eval_rewards, ew_rewards],
        tick_labels=["Random", "GFlowNet", "Equal-Weight"],
    )
    plt.grid()
    plt.title("Box Plot — Pareto Reward by Method (higher is better)")
    plt.xlabel("Method")
    plt.ylabel("Pareto-based Reward")
    _save("09_reward_boxplot.png")


def plot_metric_per_trial(rand_vals, eval_vals, label, ew_val=None):
    """Line plot of a metric (Sharpe or MDD) per trial.

    EW's metric is fixed (deterministic portfolio), shown as a horizontal line.
    """
    nx = np.arange(len(rand_vals))
    plt.plot(nx, rand_vals, color="lightblue", label=f"Random {label}")
    plt.plot(nx, eval_vals, color="darkblue",  label=f"GFlowNet {label}")
    if ew_val is not None:
        plt.axhline(ew_val, color="orange", linestyle="--", linewidth=1.5,
                    label=f"Equal-Weight {label}")
    plt.grid()
    plt.ylabel(label)
    plt.xlabel("Trial")
    plt.legend(fontsize=8)
    plt.title(f"Random vs GFlowNet vs Equal-Weight — {label}")
    _save(f"10_metric_{label.lower()}.png")


def plot_reward_lines(rand_rewards, eval_rewards, ew_rewards):
    """Line plot of per-trial mean Pareto reward for all three methods."""
    nx = np.arange(len(rand_rewards))
    plt.plot(nx, rand_rewards, color="lightblue", marker='o', markersize=3,
             linewidth=1, alpha=0.7, label="Random")
    plt.plot(nx, eval_rewards, color="darkblue",  marker='o', markersize=3,
             linewidth=1, alpha=0.7, label="GFlowNet")
    plt.plot(nx, ew_rewards,   color="orange",    marker='D', markersize=3,
             linewidth=1, alpha=0.7, label="Equal-Weight")
    plt.grid()
    plt.ylabel("Pareto-based Reward")
    plt.xlabel("Trial")
    plt.title("Random vs GFlowNet vs Equal-Weight (higher is better)")
    plt.legend(fontsize=8)
    _save("11_reward_lines.png")


# ---------------------------------------------------------------------------
# Final summary table (publication-style figure + printed report)
# ---------------------------------------------------------------------------

def plot_final_summary(rand_sharpe, rand_mdd, rand_reward,
                       eval_sharpe, eval_mdd, eval_reward,
                       ew_sharpe, ew_mdd, ew_reward,
                       dr_vs_ew=None, dr_vs_rand=None):
    """Render a summary table as a figure and print a full statistical report.

    For each metric (Sharpe, MDD, Pareto Reward) and each method, computes:
      - Mean and Std across 50 trials
      - 95% confidence interval (t-distribution)
      - Welch t-test p-value for GFlowNet vs Random

    EW is deterministic so its std/CI are computed from the joint-ranking
    variation across trials (its Sharpe/MDD are fixed but its Pareto reward
    varies because the Pareto ranking changes with each GFN sample batch).

    Args:
        rand_sharpe, rand_mdd, rand_reward: (n_trials,) arrays for Random.
        eval_sharpe, eval_mdd, eval_reward: (n_trials,) arrays for GFlowNet.
        ew_sharpe, ew_mdd, ew_reward: scalar or (n_trials,) for Equal-Weight.
        dr_vs_ew: (n_trials,) array of dominance rates vs EW per trial.
        dr_vs_rand: (n_trials,) array of dominance rates vs Random mean per trial.
    """
    n = len(rand_reward)

    def _ci95(arr):
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        h = stats.t.ppf(0.975, df=len(arr) - 1) * se
        return arr.mean() - h, arr.mean() + h

    def _fmt_ci(lo, hi):
        return f"[{lo:.4f}, {hi:.4f}]"

    # EW Sharpe/MDD are deterministic scalars; broadcast to arrays for uniform handling
    ew_sharpe_arr = np.full(n, ew_sharpe) if np.ndim(ew_sharpe) == 0 else np.asarray(ew_sharpe)
    ew_mdd_arr = np.full(n, ew_mdd) if np.ndim(ew_mdd) == 0 else np.asarray(ew_mdd)
    ew_reward_arr = np.asarray(ew_reward)

    metrics = [
        ("Sharpe Ratio",   rand_sharpe,  eval_sharpe,  ew_sharpe_arr,  "higher"),
        ("Max Drawdown",   rand_mdd,     eval_mdd,     ew_mdd_arr,     "lower"),
        ("Pareto Reward",  rand_reward,  eval_reward,  ew_reward_arr,  "higher"),
    ]

    # ── Printed report ──
    sep = "=" * 90
    print(f"\n{sep}")
    print("FINAL TEST RESULTS — 50-Trial Statistical Summary")
    print(sep)
    print(f"  {'Metric':<18} {'Method':<14} {'Mean':>9} {'Std':>9} {'95% CI':>22}  {'p-value':>9}")
    print("-" * 90)

    for metric_name, r_arr, g_arr, e_arr, direction in metrics:
        t_stat, p_val = stats.ttest_ind(g_arr, r_arr, equal_var=False)
        # One-sided: is GFN better than Random in the expected direction?
        if direction == "higher":
            p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
        else:
            p_one = p_val / 2 if t_stat < 0 else 1 - p_val / 2

        for label, arr, show_p in [("Random", r_arr, False),
                                    ("GFlowNet", g_arr, True),
                                    ("Equal-Weight", e_arr, False)]:
            lo, hi = _ci95(arr)
            p_str = f"{p_one:.2e}" if show_p else ""
            print(f"  {metric_name:<18} {label:<14} {arr.mean():9.4f} {arr.std(ddof=1):9.4f} {_fmt_ci(lo, hi):>22}  {p_str:>9}")
        print("-" * 90)

    # ── Dominance rates ──
    if dr_vs_ew is not None:
        dr_vs_ew = np.asarray(dr_vs_ew)
        dr_vs_rand = np.asarray(dr_vs_rand)
        lo_ew, hi_ew = _ci95(dr_vs_ew)
        lo_rd, hi_rd = _ci95(dr_vs_rand)
        print(f"  {'% GFN > EW':<18} {'GFlowNet':<14} {dr_vs_ew.mean():8.1f}% {dr_vs_ew.std(ddof=1):8.1f}% {f'[{lo_ew:.1f}%, {hi_ew:.1f}%]':>22}")
        print(f"  {'% GFN > Rand mean':<18} {'GFlowNet':<14} {dr_vs_rand.mean():8.1f}% {dr_vs_rand.std(ddof=1):8.1f}% {f'[{lo_rd:.1f}%, {hi_rd:.1f}%]':>22}")
        print("-" * 90)

    print(f"  p-values: one-sided Welch t-test (GFlowNet vs Random)")
    print(f"  Dominance: % of GFN portfolios with higher Sharpe AND lower MDD than reference")
    print(f"  Trials: {n}")
    print(sep)

    # ── Table figure ──
    row_labels = []
    cell_text = []

    for metric_name, r_arr, g_arr, e_arr, direction in metrics:
        t_stat, p_val = stats.ttest_ind(g_arr, r_arr, equal_var=False)
        if direction == "higher":
            p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
        else:
            p_one = p_val / 2 if t_stat < 0 else 1 - p_val / 2

        row_labels.append(metric_name)
        row = []
        for arr in [r_arr, g_arr, e_arr]:
            lo, hi = _ci95(arr)
            row.append(f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}\n{_fmt_ci(lo, hi)}")
        row.append(f"{p_one:.2e}")
        cell_text.append(row)

    # Add dominance rate rows
    if dr_vs_ew is not None:
        lo_ew, hi_ew = _ci95(dr_vs_ew)
        lo_rd, hi_rd = _ci95(dr_vs_rand)
        row_labels.append("% GFN > EW")
        cell_text.append([
            "—",
            f"{dr_vs_ew.mean():.1f}% ± {dr_vs_ew.std(ddof=1):.1f}%\n[{lo_ew:.1f}%, {hi_ew:.1f}%]",
            "—",
            "—",
        ])
        row_labels.append("% GFN > Rand mean")
        cell_text.append([
            "—",
            f"{dr_vs_rand.mean():.1f}% ± {dr_vs_rand.std(ddof=1):.1f}%\n[{lo_rd:.1f}%, {hi_rd:.1f}%]",
            "—",
            "—",
        ])

    col_labels = ["Random", "GFlowNet", "Equal-Weight", "p-value\n(GFN vs Rand)"]

    n_rows = len(row_labels)
    fig_height = 2.0 + n_rows * 0.8
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    ax.set_title("Final Test Results — 50-Trial Statistical Summary",
                 fontsize=13, fontweight="bold", pad=20)

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.4)

    # Style header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style row labels
    for i in range(n_rows):
        table[i + 1, -1].set_facecolor("#D6E4F0")
        table[i + 1, -1].set_text_props(fontweight="bold")

    # Highlight GFlowNet column
    for i in range(n_rows):
        table[i + 1, 1].set_facecolor("#E2EFDA")

    _save("99_final_summary.png")
