import torch

from src.config import N_ITER, BATCH, LAMBDA_AUX, AUX_BATCH


def train(gfn, env, sampler, optimizer, n_iter=N_ITER, batch=BATCH):
    lossarray = []
    for i in range(n_iter):
        trajectories = sampler.sample_trajectories(env, batch)
        optimizer.zero_grad()
        loss = gfn.loss(env, trajectories)
        loss.backward()
        optimizer.step()
        lossarray.append(loss.item())
        print(f"Iteration {i+1}/{n_iter}, Loss: {loss.item():.4f}, LogZ: {gfn.logZ.item():.4f}")
    return lossarray


def train_lggfn(gfn_main, gfn_aux, env, aux_env, sampler_main, sampler_aux,
                optimizer_main, optimizer_aux,
                lambda_aux=LAMBDA_AUX, n_iter=N_ITER,
                batch=BATCH, aux_batch=AUX_BATCH):
    """Train the main GFlowNet with a Loss-Guided auxiliary agent (LGGFN).

    Implements Algorithm 1 from Malek et al. (2025) "Loss-Guided Auxiliary
    Agents for Overcoming Mode Collapse in GFlowNets":

      Each iteration:
        1. Sample tau_aux ~ F_aux and tau_main ~ F_main.
        2. Compute the main model's TB loss on tau_aux (no gradient).
        3. Set aux_env reward bonus = lambda_aux * loss_main_on_aux.
        4. Update F_aux with the augmented reward (R_aux = R + lambda * L_main).
        5. Update F_main on the union of tau_main and tau_aux with reward R.

    Args:
        gfn_main: Main TBGFlowNet.
        gfn_aux:  Auxiliary TBGFlowNet (separate backbone from gfn_main).
        env:      PortfolioEnv used for main training and evaluation.
        aux_env:  AuxPortfolioEnv whose log_reward adds the loss bonus.
        sampler_main: Sampler backed by gfn_main's forward policy.
        sampler_aux:  Sampler backed by gfn_aux's forward policy.
        optimizer_main: Optimiser for gfn_main parameters.
        optimizer_aux:  Optimiser for gfn_aux parameters.
        lambda_aux: Weight of the loss bonus in the auxiliary reward.
        n_iter:    Number of training iterations.
        batch:     Trajectories per step for the main sampler.
        aux_batch: Trajectories per step for the auxiliary sampler.

    Returns:
        List of per-iteration main-model losses.
    """
    lossarray = []

    for i in range(n_iter):
        # --- 1. Sample trajectories from both policies ---
        tau_aux = sampler_aux.sample_trajectories(env, aux_batch)
        tau_main = sampler_main.sample_trajectories(env, batch)

        # --- 2. Compute main model's loss on auxiliary trajectories (signal only) ---
        with torch.no_grad():
            loss_main_on_aux = gfn_main.loss(env, tau_aux).item()

        # --- 3 & 4. Update auxiliary GFlowNet with augmented reward ---
        aux_env.set_loss_bonus(lambda_aux * loss_main_on_aux)
        optimizer_aux.zero_grad()
        loss_aux = gfn_aux.loss(aux_env, tau_aux)
        loss_aux.backward()
        optimizer_aux.step()

        # --- 5. Update main GFlowNet on the union of tau_main and tau_aux ---
        optimizer_main.zero_grad()
        loss_main = (gfn_main.loss(env, tau_main) + gfn_main.loss(env, tau_aux)) / 2
        loss_main.backward()
        optimizer_main.step()

        lossarray.append(loss_main.item())
        print(f"Iteration {i+1}/{n_iter}, "
              f"Loss: {loss_main.item():.4f}, "
              f"LogZ: {gfn_main.logZ.item():.4f}, "
              f"AuxLoss: {loss_aux.item():.4f}, "
              f"LossBonus: {lambda_aux * loss_main_on_aux:.4f}")

    return lossarray
