import numpy as np
import torch

from gfn.env import DiscreteEnv
from gfn.states import DiscreteStates

from src.utils import paretorank, reward as pareto_reward


class PortfolioEnv(DiscreteEnv):
    def __init__(self, num, maxassets, maxweight, stepsize, returnsdata):
        self.num = num
        self.maxassets = maxassets
        self.maxweight = maxweight
        self.stepsize = stepsize
        self.numweights = int(maxweight / stepsize)
        self.n_actions = num * self.numweights + 1
        self.s0 = torch.zeros(num)
        self.sf = torch.full((num,), -float('inf'))
        self.returns = torch.FloatTensor(returnsdata)
        self._ref_sharpe = None
        self._ref_mdd = None

        self.action_shape = (1,)
        self.dummy_action = torch.tensor([-1])
        self.exit_action = torch.tensor([self.n_actions - 1])
        self.state_shape = (num,)
        super().__init__(self.n_actions, self.s0, (num,))

    def set_reference_portfolio(self, weights):
        """Pre-compute and cache the reference portfolio's Sharpe and MDD.

        When set, log_reward includes this reference point in the Pareto
        ranking so the GFlowNet must compete against it during training.
        The reference's own reward is stripped before returning.
        """
        data_np = self.returns.numpy()
        ref_returns = data_np @ weights
        self._ref_sharpe = float(np.sqrt(252) * ref_returns.mean() / ref_returns.std(ddof=1))
        ref_wealth = np.exp(np.cumsum(ref_returns))
        ref_peak = np.maximum.accumulate(ref_wealth)
        self._ref_mdd = float(-(ref_wealth / ref_peak - 1.0).min())

    def sharpe(self, r):
        mu = r.mean()
        sd = r.std(ddof=1)
        return float(np.sqrt(252)) * mu / sd

    def mdd(self, r):
        wealth = np.exp(np.cumsum(r))
        peak = np.maximum.accumulate(wealth)
        dd = wealth / peak - 1.0
        return float(-dd.min())

    def evaluate_portfolio(self, w, data):
        active = np.nonzero(w)[0]
        val = data[:, active]
        r = np.matmul(val, w[active])
        return {
            "sharpe": self.sharpe(r),
            "mdd": self.mdd(r),
        }

    def log_reward(self, final_state):
        weights = final_state.tensor.numpy()
        data_np = self.returns.numpy()

        portfolio_returns = data_np @ weights.T

        mu = portfolio_returns.mean(axis=0)
        sd = portfolio_returns.std(axis=0, ddof=1)
        sharpes = np.sqrt(252) * mu / sd

        wealth = np.exp(np.cumsum(portfolio_returns, axis=0))
        peak = np.maximum.accumulate(wealth, axis=0)
        dd = (wealth / peak) - 1.0
        mdds = -dd.min(axis=0)

        # Include reference portfolio in Pareto ranking if set
        if self._ref_sharpe is not None:
            all_sharpes = np.append(sharpes, self._ref_sharpe)
            all_mdds = np.append(mdds, self._ref_mdd)
            ranks = paretorank(all_sharpes, all_mdds)
            ranks = ranks[:-1]  # strip the reference point's rank
        else:
            ranks = paretorank(sharpes, mdds)

        rewards = pareto_reward(ranks)

        return torch.log(torch.FloatTensor(rewards))

    def step(self, states, actions):
        new = states.clone()
        batch_size = new.tensor.shape[0]

        if hasattr(actions, 'tensor'):
            actions_tensor = actions.tensor.squeeze(-1)
        else:
            actions_tensor = actions.squeeze(-1) if actions.dim() > 1 else actions

        nonexit = actions_tensor < self.n_actions - 1

        if nonexit.any():
            nonexactions = (actions_tensor)[nonexit]
            indices = nonexactions // self.numweights
            weight = ((nonexactions % self.numweights) + 1) * self.stepsize

            batchindex = torch.arange(batch_size)[nonexit]

            new.tensor[batchindex, indices] = weight

        return new

    def backward_step(self, states, actions):
        new = states.clone()
        batch_size = new.shape[0]

        if hasattr(actions, 'tensor'):
            actions_tensor = actions.tensor.squeeze(-1)
        else:
            actions_tensor = actions.squeeze(-1) if actions.dim() > 1 else actions

        asset_indices = actions_tensor // self.numweights
        batchindex = torch.arange(batch_size)
        new.tensor[batchindex, asset_indices] = 0.0

        return new

    def update_masks(self, states):
        statetensor = states.tensor

        batch_size = statetensor.shape[0]
        selected = statetensor > 0
        n = selected.sum(dim=1)
        current = statetensor.sum(dim=1)

        states.forward_masks = torch.zeros(batch_size, self.n_actions, dtype=torch.bool)
        states.backward_masks = torch.zeros(batch_size, self.n_actions - 1, dtype=torch.bool)

        for action in range(self.n_actions - 1):
            assetid = action // self.numweights
            weight = ((action % self.numweights) + 1) * self.stepsize

            notselected = ~selected[:, assetid]
            validweight = weight <= self.maxweight
            notexceed = (current + weight) <= 1 + 1e-6
            notmax = n < self.maxassets

            states.forward_masks[:, action] = (notselected) & (validweight) & (notexceed) & (notmax)

        for asset_idx in range(self.num):
            start_action = asset_idx * self.numweights
            end_action = start_action + self.numweights
            states.backward_masks[:, start_action:end_action] = selected[:, asset_idx].unsqueeze(1)

        sum_is_one = torch.isclose(current, torch.tensor(1.0), atol=1e-6)
        atmax = (n >= self.maxassets)
        states.forward_masks[:, -1] = sum_is_one | atmax

    def make_States_class(self):
        env = self

        class PortfolioStates(DiscreteStates):
            def make_random_states_tensor(self, batchshape):
                return env.s0.repeat(*batchshape, 1)

            def update_masks(self):
                env.update_masks(self)

        return PortfolioStates


class AuxPortfolioEnv(PortfolioEnv):
    """PortfolioEnv for the auxiliary GFlowNet (LGGFN).

    Identical to PortfolioEnv except that log_reward adds a scalar bonus
    equal to lambda_aux * loss_main, where loss_main is the TB loss of the
    main GFlowNet on the current batch of auxiliary trajectories.  This
    bonus pushes the auxiliary to explore regions where the main model
    currently has high loss, as described in Algorithm 1 of Malek et al.
    (2025) "Loss-Guided Auxiliary Agents for Overcoming Mode Collapse in
    GFlowNets".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_bonus = 0.0

    @classmethod
    def from_env(cls, env):
        """Construct an AuxPortfolioEnv with the same parameters as *env*."""
        aux = cls(env.num, env.maxassets, env.maxweight, env.stepsize,
                  env.returns.detach().numpy())
        aux._ref_sharpe = env._ref_sharpe
        aux._ref_mdd = env._ref_mdd
        return aux

    def set_loss_bonus(self, bonus):
        """Set the per-step loss bonus (call before computing the aux loss)."""
        self._loss_bonus = float(bonus)

    def log_reward(self, final_state):
        return super().log_reward(final_state) + self._loss_bonus
