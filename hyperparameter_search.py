#!/usr/bin/env python3
"""
hyperparameter_search.py

Bayesian hyperparameter search for the GFlowNet portfolio-optimisation model.
Uses Optuna (TPE sampler) to efficiently explore the search space.

Validation setup
----------------
Each trial trains on the expanding-window fold ending on SEARCH_TRAIN_END
(2022-12-31) and evaluates on the held-out dev period 2023-01-01 – 2023-08-31.
This keeps the final test set (2023-09-01 onward) completely unseen.
The objective is the mean Pareto reward of SEARCH_N_SAMPLES portfolios sampled
from the trained policy.

Search space
------------
  k              – max assets per portfolio         [3, 15]
  maxweight      – max weight per asset             [0.15, 0.60]
  lr_policy      – policy learning rate             [1e-5, 1e-2]  log-scale
  lr_logz        – logZ learning rate               [1e-4, 5e-2]  log-scale
  n_iter         – training iterations per trial    [50, 150]
  batch          – main trajectories / step         {4, 6, 8, 12, 16, 24}
  aux_batch      – auxiliary trajectories / step    {2, 4, 6, 8, 12}
  lambda_aux     – aux loss-bonus weight            [0.05, 0.80]
  n_hidden_layers– MLP depth (PF and PB)            [2, 6]

Usage
-----
    python hyperparameter_search.py                          # 60 trials
    python hyperparameter_search.py --n-trials 120 --timeout 7200
    python hyperparameter_search.py --n-trials 100 --update-config

After the search:
  • All results are written to evals/hypersearch_results.json (sorted by score).
  • The best configuration is printed to stdout.
  • Pass --update-config to automatically overwrite src/config.py.

Dependencies
------------
    pip install optuna          (not in requirements.txt by default)
"""

import argparse
import contextlib
import io
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Search-phase constants (separate from src/config.py)
# ---------------------------------------------------------------------------

SEARCH_N_ITER    = 150    # GFlowNet training iterations cap per trial
SEARCH_N_SAMPLES = 300    # portfolios sampled for dev-set evaluation per trial
SEARCH_TRAIN_END = "2022-12-31"
SEARCH_DEV_START = "2023-01-01"
SEARCH_DEV_END   = "2023-08-31"
RESULTS_JSON     = "evals/hypersearch_results.json"

# ---------------------------------------------------------------------------
# Data loading — downloaded once and reused across all trials
# ---------------------------------------------------------------------------

_DATA_CACHE: tuple | None = None


def get_search_data() -> tuple[np.ndarray, np.ndarray]:
    """Return (train_np, dev_np) log-return arrays for the search fold.

    Downloads from Yahoo Finance on first call; subsequent calls use the
    in-process cache so market data is only fetched once per search run.
    """
    global _DATA_CACHE
    if _DATA_CACHE is None:
        from src.data import load_data
        print("Downloading / loading market data …")
        logreturn = load_data()
        train_np = (
            logreturn.loc[:SEARCH_TRAIN_END]
            .dropna(how="all")
            .values.astype(np.float32)
        )
        dev_np = (
            logreturn.loc[SEARCH_DEV_START:SEARCH_DEV_END]
            .dropna(how="all")
            .values.astype(np.float32)
        )
        _DATA_CACHE = (train_np, dev_np)
    return _DATA_CACHE


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial, train_data: np.ndarray, dev_data: np.ndarray) -> float:
    """Train a GFlowNet with trial params and return mean dev-set Pareto reward."""
    import optuna
    from gfn.gflownet import TBGFlowNet
    from gfn.modules import DiscretePolicyEstimator
    from gfn.samplers import Sampler
    from gfn.utils.modules import MLP
    from torch.optim import Adam

    from src.config import N, STEP
    from src.environment import AuxPortfolioEnv, PortfolioEnv
    from src.evaluate import returnval, sample_gfn_portfolios
    from src.train import train_lggfn

    # ---- 1. Suggest hyperparameters ----------------------------------------
    k           = trial.suggest_int("k",              3,    15)
    maxweight   = trial.suggest_float("maxweight",   0.15,  0.60, step=0.05)
    lr_policy   = trial.suggest_float("lr_policy",  1e-5,  1e-2, log=True)
    lr_logz     = trial.suggest_float("lr_logz",    1e-4,  5e-2, log=True)
    n_iter      = trial.suggest_int("n_iter",        50,    SEARCH_N_ITER, step=10)
    batch       = trial.suggest_categorical("batch", [4, 6, 8, 12, 16, 24])
    aux_batch   = trial.suggest_categorical("aux_batch", [2, 4, 6, 8, 12])
    lambda_aux  = trial.suggest_float("lambda_aux",  0.05,  0.80)
    n_hidden    = trial.suggest_int("n_hidden_layers", 2,   6)

    # Feasibility check: it must be possible for weights to sum to 1.0
    # (at least one valid portfolio must be constructable).
    if maxweight * k < 1.0:
        raise optuna.exceptions.TrialPruned()

    # ---- 2. Build environments ---------------------------------------------
    ew_w = np.ones(N) / N
    env     = PortfolioEnv(N, k, maxweight, STEP, train_data)
    env.set_reference_portfolio(ew_w)
    aux_env = AuxPortfolioEnv.from_env(env)

    # ---- 3. Build GFlowNet models with tunable MLP depth ------------------
    def _build_gfn(ref_env):
        pf_mod = MLP(N, ref_env.n_actions,     n_hidden_layers=n_hidden)
        pb_mod = MLP(N, ref_env.n_actions - 1, trunk=pf_mod.trunk,
                     n_hidden_layers=n_hidden)
        pf  = DiscretePolicyEstimator(pf_mod, ref_env.n_actions)
        pb  = DiscretePolicyEstimator(pb_mod, ref_env.n_actions, is_backward=True)
        gfn = TBGFlowNet(pf, pb)
        sampler = Sampler(pf)
        all_params = dict(gfn.named_parameters())
        non_logz   = [v for pname, v in all_params.items() if pname != "logZ"]
        opt = Adam(non_logz, lr=lr_policy)
        opt.add_param_group({"params": [all_params["logZ"]], "lr": lr_logz})
        return gfn, sampler, opt

    gfn_main, sampler_main, opt_main = _build_gfn(env)
    gfn_aux,  sampler_aux,  opt_aux  = _build_gfn(aux_env)

    # ---- 4. Train — verbose output suppressed for cleaner search logs -----
    with contextlib.redirect_stdout(io.StringIO()), \
         warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_lggfn(
            gfn_main, gfn_aux, env, aux_env,
            sampler_main, sampler_aux,
            opt_main, opt_aux,
            lambda_aux=lambda_aux,
            n_iter=n_iter,
            batch=batch,
            aux_batch=aux_batch,
        )

    # ---- 5. Sample portfolios and evaluate on the dev period --------------
    portfolios = sample_gfn_portfolios(sampler_main, env, SEARCH_N_SAMPLES)
    df         = returnval(dev_data, portfolios)

    mean_reward = float(df["reward"].mean())
    mean_sharpe = float(df["sharpe"].mean())
    mean_mdd    = float(df["mdd"].mean())
    pct_rank0   = float((df["rank"] == 0).mean())

    # Store secondary metrics as user attributes for later inspection
    trial.set_user_attr("mean_sharpe", mean_sharpe)
    trial.set_user_attr("mean_mdd",    mean_mdd)
    trial.set_user_attr("pct_rank0",   pct_rank0)
    trial.set_user_attr("mean_reward", mean_reward)

    return mean_reward


# ---------------------------------------------------------------------------
# Config update helper
# ---------------------------------------------------------------------------

def update_config(best_params: dict) -> None:
    """Overwrite relevant constants in src/config.py with the best params found."""
    import re

    # Map config variable name → (optuna param key, formatter)
    replacements = {
        "K":          ("k",           str),
        "MAXWEIGHT":  ("maxweight",   lambda v: f"{v:.2f}"),
        "BATCH":      ("batch",       str),
        "N_ITER":     ("n_iter",      str),
        "LR_POLICY":  ("lr_policy",   lambda v: f"{v:.6g}"),
        "LR_LOGZ":    ("lr_logz",     lambda v: f"{v:.6g}"),
        "LAMBDA_AUX": ("lambda_aux",  lambda v: f"{v:.4f}"),
        "AUX_BATCH":  ("aux_batch",   str),
    }

    config_path = "src/config.py"
    with open(config_path) as fh:
        src = fh.read()

    for cfg_var, (param_key, fmt) in replacements.items():
        if param_key not in best_params:
            continue
        val_str = fmt(best_params[param_key])
        src = re.sub(
            rf"^({cfg_var}\s*=\s*).*$",
            rf"\g<1>{val_str}",
            src,
            flags=re.MULTILINE,
        )

    with open(config_path, "w") as fh:
        fh.write(src)

    print(f"src/config.py updated with best hyperparameters.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global SEARCH_N_ITER, SEARCH_N_SAMPLES

    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter search for the GFlowNet portfolio model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-trials", type=int, default=60,
        help="Number of Optuna trials to run",
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Optional wall-clock time budget in seconds (overrides --n-trials)",
    )
    parser.add_argument(
        "--output-json", type=str, default=RESULTS_JSON,
        help="Path to write full results JSON",
    )
    parser.add_argument(
        "--update-config", action="store_true",
        help="Overwrite src/config.py with the best params after the search",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the Optuna TPE sampler",
    )
    parser.add_argument(
        "--search-n-iter", type=int, default=SEARCH_N_ITER,
        help="Max GFlowNet training iterations per trial (lower = faster search)",
    )
    parser.add_argument(
        "--search-n-samples", type=int, default=SEARCH_N_SAMPLES,
        help="Number of portfolios sampled per trial for dev-set evaluation",
    )
    args = parser.parse_args()

    # Override module-level constants if user supplied flags
    SEARCH_N_ITER    = args.search_n_iter
    SEARCH_N_SAMPLES = args.search_n_samples

    # --- verify optuna is available -----------------------------------------
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        sys.exit(
            "\noptuna is not installed.  Install it with:\n"
            "    pip install optuna\n"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # --- load data once, shared across all trials ---------------------------
    train_data, dev_data = get_search_data()
    print(f"Train : {train_data.shape[0]} days, {train_data.shape[1]} assets  "
          f"(up to {SEARCH_TRAIN_END})")
    print(f"Dev   : {dev_data.shape[0]} days, {dev_data.shape[1]} assets  "
          f"({SEARCH_DEV_START} – {SEARCH_DEV_END})")
    print(f"\nSearch budget  : {args.n_trials} trials, timeout={args.timeout}s")
    print(f"Iters per trial: up to {SEARCH_N_ITER}")
    print(f"Samples / trial: {SEARCH_N_SAMPLES}\n")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    # --- create and run Optuna study ----------------------------------------
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=args.seed, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: objective(trial, train_data, dev_data),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
        catch=(Exception,),   # log failures without crashing the whole search
    )

    # --- report best trial --------------------------------------------------
    if not study.best_trial:
        print("No trials completed successfully.")
        return study

    best = study.best_trial
    print("\n" + "=" * 62)
    print("BEST TRIAL")
    print("=" * 62)
    print(f"  Trial #        : {best.number}")
    print(f"  Mean reward    : {best.value:.4f}")
    print()
    for pname, pval in best.params.items():
        print(f"  {pname:22s}: {pval}")
    print()
    ua = best.user_attrs
    print(f"  mean_sharpe    : {ua.get('mean_sharpe', float('nan')):.4f}")
    print(f"  mean_mdd       : {ua.get('mean_mdd',    float('nan')):.4f}")
    print(f"  pct_rank0      : {ua.get('pct_rank0',   float('nan')):.2%}")
    print("=" * 62)

    # --- save all completed trials ------------------------------------------
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    rows = []
    for t in sorted(completed, key=lambda x: x.value or 0, reverse=True):
        row = {"trial": t.number, "value": t.value}
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)

    output = {
        "timestamp":    datetime.now().isoformat(),
        "n_trials":     len(study.trials),
        "n_completed":  len(completed),
        "best_trial":   best.number,
        "best_value":   best.value,
        "best_params":  best.params,
        "best_metrics": best.user_attrs,
        "search_config": {
            "search_n_iter":    SEARCH_N_ITER,
            "search_n_samples": SEARCH_N_SAMPLES,
            "train_end":        SEARCH_TRAIN_END,
            "dev_start":        SEARCH_DEV_START,
            "dev_end":          SEARCH_DEV_END,
        },
        "all_trials":   rows,
    }

    with open(args.output_json, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nFull results written to: {args.output_json}")

    # --- optionally update src/config.py ------------------------------------
    if args.update_config:
        update_config(best.params)

    return study


if __name__ == "__main__":
    main()
