import numpy as np
import pandas as pd

from src.config import NUMSAMPLES
from src.data import TICKERS
from src.utils import evaluate_portfolio, paretorank, reward


def ew_portfolio(n_assets):
    """Equal-weight portfolio: 1/N weight in each of the N assets.

    With N=100 and STEP=0.01 this places exactly 1% in every stock,
    which satisfies the weight-granularity and max-weight constraints.
    Note: this uses all N assets, not the GFlowNet's K-asset limit.
    """
    return np.ones(n_assets) / n_assets


def sample_gfn_portfolios(sampler, env, n_samples):
    """Extract terminal portfolio weight vectors from a trained GFlowNet.

    Samples *n_samples* trajectories, finds each trajectory's terminal state
    (the last non-sink state), and returns them as an (n_samples, N) array.
    """
    sampled = sampler.sample_trajectories(env, n_samples)
    states = sampled.states.tensor
    terminal_indices = (states != env.sf[0]).all(dim=-1).sum(dim=0) - 1
    portfolios = [states[terminal_indices[i], i].numpy() for i in range(n_samples)]
    return np.array(portfolios)


def totaleval(data, sampler, n_samples=None):
    if n_samples is None:
        n_samples = NUMSAMPLES
    sharpes, mdds = [], []
    for i in range(n_samples):
        w = sampler(TICKERS)
        portfolioval = evaluate_portfolio(w, data)
        sharpes.append(portfolioval["sharpe"])
        mdds.append(portfolioval["mdd"])

    ranks = paretorank(sharpes, mdds)
    rewards = reward(ranks)

    table = pd.DataFrame({"sharpe": sharpes, "mdd": mdds, "rank": ranks, "reward": rewards})
    return table


def returnval(data, portfolios):
    sharpes, mdds = [], []
    for portfolio in portfolios:
        val = evaluate_portfolio(portfolio, data)
        sharpes.append(val["sharpe"])
        mdds.append(val["mdd"])

    ranks = paretorank(sharpes, mdds)
    rewards = reward(ranks)

    table = pd.DataFrame({"sharpe": sharpes, "mdd": mdds, "rank": ranks, "reward": rewards})
    return table


def dominance_rate(gfn_df, ref_sharpe, ref_mdd):
    """Percentage of GFN portfolios that strictly dominate a reference point.

    A portfolio dominates the reference if it has higher Sharpe AND lower MDD.
    """
    beats = (gfn_df["sharpe"] > ref_sharpe) & (gfn_df["mdd"] < ref_mdd)
    return float(beats.mean()) * 100
