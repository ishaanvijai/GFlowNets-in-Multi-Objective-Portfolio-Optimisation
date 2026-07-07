# GFlowNets in Multi-Objective Portfolio Optimisation

Applying Generative Flow Networks (GFlowNets) to the problem of constructing diverse, high-quality equity portfolios that trade off Sharpe ratio against Maximum Drawdown.

---

## Overview

Standard portfolio optimisers return a single "optimal" solution. GFlowNets, by contrast, learn a generative policy that samples solutions proportionally to their reward — producing a diverse set of portfolios that explore the Pareto frontier between two competing objectives:

- **Sharpe Ratio** — annualised return-to-risk (higher is better)
- **Maximum Drawdown (MDD)** — worst peak-to-trough loss over the period (lower is better)

Rather than finding one portfolio, the trained GFlowNet samples many portfolios that collectively span the trade-off surface, enabling a richer picture of the risk-return landscape.

---

## Methodology

### Data
- **Universe:** 100 large-cap US equities (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, and 93 others spanning technology, healthcare, consumer, energy, and financials)
- **Source:** Yahoo Finance via `yfinance`; 10 years of daily close prices (2015-09-01 to 2025-09-01)
- **Returns:** Log returns computed as `log(P_t / P_{t-1})`
- **Split:** Train — up to 2023-08-31; Test — 2023-09-01 onwards (~2 years out-of-sample)

### Portfolio Constraints
- Maximum **5 assets** per portfolio
- Weights in **1% increments** (step = 0.01)
- Maximum **30% weight** for any single asset
- Weights must sum exactly to **100%**

### GFlowNet Formulation
- **State:** A 100-dimensional weight vector representing allocation to each asset (initialised to zeros)
- **Actions:** Assign a weight level to an asset (100 assets × 30 weight levels = 3000 actions) or EXIT (1 action) — 3001 actions total
- **Termination:** EXIT action is enabled when weights sum to 1.0 or the maximum number of assets is reached
- **Objective:** Trajectory Balance (TB) — trains the policy to sample complete portfolios with probability proportional to their reward
- **Reward signal:** Sharpe-based log reward during training; post-hoc Pareto ranking used for multi-objective evaluation

### Pareto Ranking
After sampling, portfolios are ranked by non-dominated Pareto fronts across both objectives. Rank 0 = the Pareto-optimal front (highest Sharpe AND lowest MDD simultaneously); higher ranks are progressively dominated. This ranking is used to evaluate the diversity and quality of the generated portfolios.

---

## Model Architecture

| Component | Details |
|---|---|
| Forward policy (PF) | 4-layer MLP, input 100 → output 3001 |
| Backward policy (PB) | 4-layer MLP, shares trunk with PF, output 3000 |
| GFlowNet objective | Trajectory Balance (TB) |
| logZ parameter | Learned scalar, 10× higher learning rate |
| Optimiser | Adam — policy LR 0.0005, logZ LR 0.005 |
| Training | 300 iterations, batch size 12 trajectories |

---

## Results

All results compare 1000 GFlowNet-sampled portfolios against 1000 uniformly random portfolios satisfying the same constraints.

### Single Evaluation

| Metric | Random | GFlowNet | Change |
|---|---|---|---|
| Mean Sharpe (train) | 0.683 | 0.672 | -1.6% |
| Mean MDD (train) | 0.353 | **0.270** | **-23.5%** |
| Reward (S − 0.4·MDD) (train) | 0.541 | **0.564** | +4.2% |
| Mean Sharpe (test) | 0.706 | 0.685 | -3.0% |
| Mean MDD (test) | 0.191 | **0.144** | **-24.6%** |
| Reward (S − 0.4·MDD) (test) | 0.630 | 0.627 | -0.4% |

The GFlowNet consistently reduces Maximum Drawdown by ~24%, at a small cost to Sharpe. The combined reward improves on the training set and is near-parity on the held-out test set, demonstrating generalisation.

### 50-Trial Repeated Experiment

To assess robustness, the evaluation was repeated 50 times (100 portfolios per trial, test set). Results show the GFlowNet systematically produces lower-MDD portfolios across all trials, with comparable or better combined reward distributions as shown in the box plots and scatter plots.

---

## Project Structure

```
GFlowNets-in-Multi-Objective-Portfolio-Optimisation/
├── README.md
├── requirements.txt
├── proj3.ipynb          # Original full analysis notebook
├── main.py              # Entry point — runs the complete pipeline
└── src/
    ├── __init__.py
    ├── config.py        # Hyperparameters and constants
    ├── data.py          # Ticker list, data download, train/test split
    ├── utils.py         # Portfolio sampling, Sharpe, MDD, Pareto ranking
    ├── evaluate.py      # totaleval and returnval evaluation functions
    ├── environment.py   # PortfolioEnv (DiscreteEnv) and PortfolioStates
    ├── model.py         # build_model — MLP, TBGFlowNet, Sampler, optimiser
    ├── train.py         # Training loop
    └── visualise.py     # All plotting functions
```

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `torchgfn`, `yfinance`, `pandas`, `numpy`, `matplotlib`, `tqdm`

---

## Usage

Run the complete pipeline (data download → training → evaluation → plots):

```bash
python main.py
```

The original notebook with full analysis and outputs is available at `proj3.ipynb`.

---

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `N` | 100 | Number of assets in universe |
| `K` | 5 | Maximum assets per portfolio |
| `STEP` | 0.01 | Weight granularity (1% increments) |
| `MAXWEIGHT` | 0.30 | Maximum weight per asset (30%) |
| `N_ITER` | 300 | Training iterations |
| `BATCH` | 12 | Trajectories per training step |
| `LR_POLICY` | 0.0005 | Learning rate for policy parameters |
| `LR_LOGZ` | 0.005 | Learning rate for logZ |
| `NUMSAMPLES` | 1000 | Default samples per evaluation run |
