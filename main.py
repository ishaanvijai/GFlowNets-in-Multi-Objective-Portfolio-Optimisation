import numpy as np

from src.config import K, STEP, MAXWEIGHT, N_ITER, BATCH, LAMBDA_AUX, AUX_BATCH, WF_WINDOWS
from src.data import TICKERS, load_data, train_test_split, walk_forward_splits
from src.utils import sampleportfolio
from src.evaluate import totaleval, returnval, sample_gfn_portfolios, ew_portfolio, dominance_rate
from src.environment import PortfolioEnv, AuxPortfolioEnv
from src.model import build_model, build_aux_model
from src.train import train_lggfn
from src import visualise


def main():
    # --- Data ---
    print("Downloading data...")
    logreturn = load_data()
    train_data, test_data = train_test_split(logreturn)
    N = len(TICKERS)
    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

    # Equal-weight portfolio (fixed — computed once, evaluated on each data split)
    ew_w = ew_portfolio(N)

    # --- Random baseline ---
    print("\nRunning random baseline on training data...")
    randrun = totaleval(train_data, sampleportfolio)
    print(randrun)

    # --- Baseline visualisations ---
    visualise.plot_pareto_scatter(randrun)
    visualise.plot_combined_reward(randrun)
    visualise.plot_rank_histogram(randrun)
    visualise.plot_sharpe_histogram(randrun)
    visualise.plot_mdd_histogram(randrun)

    # --- Walk-forward cross-validation ---
    # Each fold trains on an expanding window and evaluates on the following
    # year (dev).  The test set (2023-09-01 onward) is never touched here.
    # EW is ranked jointly with GFN portfolios so rewards are comparable.
    print("\n--- Walk-forward cross-validation ---")
    wf_splits = walk_forward_splits(logreturn)
    wf_gfn_rewards  = []
    wf_rand_rewards = []
    wf_ew_rewards   = []
    wf_gfn_sharpes  = []
    wf_rand_sharpes = []
    wf_ew_sharpes   = []
    wf_gfn_mdds     = []
    wf_rand_mdds    = []
    wf_ew_mdds      = []

    for fold_idx, (train_fold, dev_fold) in enumerate(wf_splits):
        train_end, dev_start, dev_end = WF_WINDOWS[fold_idx]
        print(f"\n  Fold {fold_idx + 1}/{len(wf_splits)}  "
              f"[train: → {train_end}  |  dev: {dev_start} → {dev_end}]  "
              f"({len(train_fold)} / {len(dev_fold)} days)")

        env_fold = PortfolioEnv(N, K, MAXWEIGHT, STEP, train_fold)
        env_fold.set_reference_portfolio(ew_w)
        aux_env_fold = AuxPortfolioEnv.from_env(env_fold)
        gfn_fold, sampler_fold, optimizer_fold = build_model(env_fold)
        gfn_aux_fold, sampler_aux_fold, optimizer_aux_fold = build_aux_model(env_fold)

        train_lggfn(
            gfn_fold, gfn_aux_fold, env_fold, aux_env_fold,
            sampler_fold, sampler_aux_fold,
            optimizer_fold, optimizer_aux_fold,
            lambda_aux=LAMBDA_AUX, n_iter=N_ITER, batch=BATCH, aux_batch=AUX_BATCH,
        )

        portfolios_fold = sample_gfn_portfolios(sampler_fold, env_fold, 500)
        rand_dev_eval   = totaleval(dev_fold, sampleportfolio, n_samples=500)

        # Rank GFN and EW jointly so rewards are on the same Pareto scale
        dev_eval_joint = returnval(dev_fold, np.vstack([portfolios_fold, ew_w[np.newaxis, :]]))
        ew_dev_row  = dev_eval_joint.iloc[-1]
        dev_eval    = dev_eval_joint.iloc[:-1].reset_index(drop=True)

        gfn_reward  = dev_eval["reward"].mean()
        rand_reward = rand_dev_eval["reward"].mean()
        ew_reward   = float(ew_dev_row["reward"])

        wf_gfn_rewards.append(gfn_reward)
        wf_rand_rewards.append(rand_reward)
        wf_ew_rewards.append(ew_reward)
        wf_gfn_sharpes.append(dev_eval["sharpe"].mean())
        wf_rand_sharpes.append(rand_dev_eval["sharpe"].mean())
        wf_ew_sharpes.append(float(ew_dev_row["sharpe"]))
        wf_gfn_mdds.append(dev_eval["mdd"].mean())
        wf_rand_mdds.append(rand_dev_eval["mdd"].mean())
        wf_ew_mdds.append(float(ew_dev_row["mdd"]))

        print(f"    Dev reward — Random: {rand_reward:.4f}  |  GFlowNet: {gfn_reward:.4f}  |  EW: {ew_reward:.4f}")
        print(f"    Dev Sharpe — Random: {rand_dev_eval['sharpe'].mean():.4f}  |  GFlowNet: {dev_eval['sharpe'].mean():.4f}  |  EW: {ew_dev_row['sharpe']:.4f}")
        print(f"    Dev MDD    — Random: {rand_dev_eval['mdd'].mean():.4f}  |  GFlowNet: {dev_eval['mdd'].mean():.4f}  |  EW: {ew_dev_row['mdd']:.4f}")

    print(f"\nWalk-forward CV summary ({len(wf_splits)} folds):")
    print(f"  Mean dev reward — Random: {np.mean(wf_rand_rewards):.4f}  |  GFlowNet: {np.mean(wf_gfn_rewards):.4f}  |  EW: {np.mean(wf_ew_rewards):.4f}")
    print(f"  Std  dev reward — Random: {np.std(wf_rand_rewards):.4f}  |  GFlowNet: {np.std(wf_gfn_rewards):.4f}  |  EW: {np.std(wf_ew_rewards):.4f}")
    print(f"  Mean dev Sharpe — Random: {np.mean(wf_rand_sharpes):.4f}  |  GFlowNet: {np.mean(wf_gfn_sharpes):.4f}  |  EW: {np.mean(wf_ew_sharpes):.4f}")
    print(f"  Mean dev MDD    — Random: {np.mean(wf_rand_mdds):.4f}  |  GFlowNet: {np.mean(wf_gfn_mdds):.4f}  |  EW: {np.mean(wf_ew_mdds):.4f}")
    visualise.plot_walk_forward_results(wf_rand_rewards, wf_gfn_rewards, wf_ew_rewards)

    # --- GFlowNet setup (full train data → final model) ---
    print("\nBuilding GFlowNet environment and models (full training set)...")
    env = PortfolioEnv(N, K, MAXWEIGHT, STEP, train_data)
    env.set_reference_portfolio(ew_w)
    aux_env = AuxPortfolioEnv.from_env(env)

    gfn, sampler, optimizer = build_model(env)
    gfn_aux, sampler_aux, optimizer_aux = build_aux_model(env)

    # --- Training (LGGFN) ---
    print("\nTraining with Loss-Guided auxiliary GFlowNet (LGGFN)...")
    lossarray = train_lggfn(
        gfn, gfn_aux, env, aux_env,
        sampler, sampler_aux,
        optimizer, optimizer_aux,
        lambda_aux=LAMBDA_AUX, n_iter=N_ITER, batch=BATCH, aux_batch=AUX_BATCH,
    )
    visualise.plot_loss_curve(lossarray)

    # --- Single evaluation (1000 samples) ---
    # EW is ranked jointly with GFN portfolios; its row is extracted then removed.
    print("\nEvaluating GFlowNet portfolios...")
    n_samples = 1000
    portfolios = sample_gfn_portfolios(sampler, env, n_samples)
    print(f"Sampled {len(portfolios)} portfolios")

    portfolios_with_ew = np.vstack([portfolios, ew_w[np.newaxis, :]])

    randruntrain = totaleval(train_data, sampleportfolio)
    evaltrain_joint = returnval(train_data, portfolios_with_ew)
    ew_train_row = evaltrain_joint.iloc[-1]
    evaltrain    = evaltrain_joint.iloc[:-1].reset_index(drop=True)

    print("\nEvaluation on training data:")
    print(f"  {'':20s} {'Random':>10}  {'GFlowNet':>10}  {'Equal-Weight':>12}")
    print(f"  {'Mean Sharpe':20s} {randruntrain['sharpe'].mean():10.4f}  {evaltrain['sharpe'].mean():10.4f}  {ew_train_row['sharpe']:12.4f}")
    print(f"  {'Mean MDD':20s} {randruntrain['mdd'].mean():10.4f}  {evaltrain['mdd'].mean():10.4f}  {ew_train_row['mdd']:12.4f}")
    print(f"  {'Mean Pareto reward':20s} {randruntrain['reward'].mean():10.4f}  {evaltrain['reward'].mean():10.4f}  {ew_train_row['reward']:12.4f}")

    # Dominance rates on train data
    rand_mean_sharpe_train = randruntrain["sharpe"].mean()
    rand_mean_mdd_train = randruntrain["mdd"].mean()
    dr_vs_ew_train = dominance_rate(evaltrain, float(ew_train_row["sharpe"]), float(ew_train_row["mdd"]))
    dr_vs_rand_train = dominance_rate(evaltrain, rand_mean_sharpe_train, rand_mean_mdd_train)
    print(f"  {'% GFN > EW':20s} {dr_vs_ew_train:10.1f}%")
    print(f"  {'% GFN > Rand mean':20s} {dr_vs_rand_train:10.1f}%")

    # 3-way scatter on train data
    visualise.plot_pareto_scatter_comparison(
        randruntrain, evaltrain,
        float(ew_train_row["sharpe"]), float(ew_train_row["mdd"]),
        title="Portfolio comparison — train data (Sharpe vs MDD)",
        save_name="12_pareto_scatter_train.png",
    )

    randrun = totaleval(test_data, sampleportfolio)
    evalrun_joint = returnval(test_data, portfolios_with_ew)
    ew_test_row = evalrun_joint.iloc[-1]
    evalrun     = evalrun_joint.iloc[:-1].reset_index(drop=True)

    print("\nEvaluation on test data:")
    print(f"  {'':20s} {'Random':>10}  {'GFlowNet':>10}  {'Equal-Weight':>12}")
    print(f"  {'Mean Sharpe':20s} {randrun['sharpe'].mean():10.4f}  {evalrun['sharpe'].mean():10.4f}  {ew_test_row['sharpe']:12.4f}")
    print(f"  {'Mean MDD':20s} {randrun['mdd'].mean():10.4f}  {evalrun['mdd'].mean():10.4f}  {ew_test_row['mdd']:12.4f}")
    print(f"  {'Mean Pareto reward':20s} {randrun['reward'].mean():10.4f}  {evalrun['reward'].mean():10.4f}  {ew_test_row['reward']:12.4f}")

    # Dominance rates on test data
    rand_mean_sharpe_test = randrun["sharpe"].mean()
    rand_mean_mdd_test = randrun["mdd"].mean()
    dr_vs_ew_test = dominance_rate(evalrun, float(ew_test_row["sharpe"]), float(ew_test_row["mdd"]))
    dr_vs_rand_test = dominance_rate(evalrun, rand_mean_sharpe_test, rand_mean_mdd_test)
    print(f"  {'% GFN > EW':20s} {dr_vs_ew_test:10.1f}%")
    print(f"  {'% GFN > Rand mean':20s} {dr_vs_rand_test:10.1f}%")

    # 3-way scatter on test data
    visualise.plot_pareto_scatter_comparison(
        randrun, evalrun,
        float(ew_test_row["sharpe"]), float(ew_test_row["mdd"]),
        title="Portfolio comparison — test data (Sharpe vs MDD)",
        save_name="13_pareto_scatter_test.png",
    )

    print(f"\nNB: {n_samples} random samples. GFlowNet trained for {N_ITER} iterations, sampled {n_samples} portfolios.")

    # Fixed EW metrics on test data (deterministic, used as reference lines below)
    ew_test_sharpe = float(ew_test_row["sharpe"])
    ew_test_mdd    = float(ew_test_row["mdd"])

    # --- Repeated 50-trial experiment ---
    print("\nRunning 50-trial repeated experiment on test data...")
    top    = 50
    newnum = 100

    newrandreward  = np.zeros(top)
    newevalreward  = np.zeros(top)
    newewreward    = np.zeros(top)
    newrandsharpe  = np.zeros(top)
    newevalsharpe  = np.zeros(top)
    newrandmdd     = np.zeros(top)
    newevalmdd     = np.zeros(top)
    dr_vs_ew       = np.zeros(top)
    dr_vs_rand     = np.zeros(top)

    for j in range(top):
        newrandrun     = totaleval(test_data, sampleportfolio, n_samples=newnum)
        trial_portfolios = sample_gfn_portfolios(sampler, env, newnum)

        # Joint Pareto ranking: GFN portfolios + EW
        trial_joint    = returnval(test_data, np.vstack([trial_portfolios, ew_w[np.newaxis, :]]))
        ew_trial_row   = trial_joint.iloc[-1]
        newevalrun     = trial_joint.iloc[:-1].reset_index(drop=True)

        newrandsharpe[j] = newrandrun["sharpe"].mean()
        newrandmdd[j]    = newrandrun["mdd"].mean()
        newevalsharpe[j] = newevalrun["sharpe"].mean()
        newevalmdd[j]    = newevalrun["mdd"].mean()

        newrandreward[j] = newrandrun["reward"].mean()
        newevalreward[j] = newevalrun["reward"].mean()
        newewreward[j]   = float(ew_trial_row["reward"])

        # Dominance rates for this trial
        dr_vs_ew[j]   = dominance_rate(newevalrun, float(ew_trial_row["sharpe"]), float(ew_trial_row["mdd"]))
        dr_vs_rand[j] = dominance_rate(newevalrun, newrandrun["sharpe"].mean(), newrandrun["mdd"].mean())

    print(f"\n50-trial summary (test data, {newnum} samples per trial):")
    print(f"  {'':20s} {'Random':>10}  {'GFlowNet':>10}  {'Equal-Weight':>12}")
    print(f"  {'Mean reward':20s} {newrandreward.mean():10.4f}  {newevalreward.mean():10.4f}  {newewreward.mean():12.4f}")
    print(f"  {'Std  reward':20s} {newrandreward.std():10.4f}  {newevalreward.std():10.4f}  {newewreward.std():12.4f}")
    print(f"  {'Mean Sharpe':20s} {newrandsharpe.mean():10.4f}  {newevalsharpe.mean():10.4f}  {ew_test_sharpe:12.4f}")
    print(f"  {'Mean MDD':20s} {newrandmdd.mean():10.4f}  {newevalmdd.mean():10.4f}  {ew_test_mdd:12.4f}")
    print(f"  {'% GFN > EW':20s} {dr_vs_ew.mean():10.1f}% ± {dr_vs_ew.std():5.1f}%")
    print(f"  {'% GFN > Rand mean':20s} {dr_vs_rand.mean():10.1f}% ± {dr_vs_rand.std():5.1f}%")

    # --- Results plots ---
    visualise.plot_trial_comparison(newrandreward, newevalreward, newewreward)
    visualise.plot_reward_boxplot(newrandreward, newevalreward, newewreward)
    visualise.plot_metric_per_trial(newrandsharpe, newevalsharpe, "Sharpe", ew_val=ew_test_sharpe)
    visualise.plot_metric_per_trial(newrandmdd,    newevalmdd,    "MDD",    ew_val=ew_test_mdd)
    visualise.plot_reward_lines(newrandreward, newevalreward, newewreward)

    # --- Final statistical summary ---
    visualise.plot_final_summary(
        newrandsharpe, newrandmdd, newrandreward,
        newevalsharpe, newevalmdd, newevalreward,
        ew_test_sharpe, ew_test_mdd, newewreward,
        dr_vs_ew=dr_vs_ew, dr_vs_rand=dr_vs_rand,
    )


if __name__ == "__main__":
    main()
