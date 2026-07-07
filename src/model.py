from torch.optim import Adam

from gfn.gflownet import TBGFlowNet
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils.modules import MLP

from src.config import LR_POLICY, LR_LOGZ


def build_model(env, lr_policy=LR_POLICY, lr_logz=LR_LOGZ):
    N = env.num

    module_PF = MLP(N, env.n_actions, n_hidden_layers=4)
    module_PB = MLP(N, env.n_actions - 1, trunk=module_PF.trunk, n_hidden_layers=4)

    pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions)
    pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, is_backward=True)

    gfn = TBGFlowNet(pf_estimator, pb_estimator)
    sampler = Sampler(pf_estimator)

    non_logz_params = [v for k, v in dict(gfn.named_parameters()).items() if k != "logZ"]

    optimizer = Adam(non_logz_params, lr=lr_policy)

    logz_params = [dict(gfn.named_parameters())["logZ"]]
    optimizer.add_param_group({"params": logz_params, "lr": lr_logz})

    return gfn, sampler, optimizer


def build_aux_model(env, lr_policy=LR_POLICY, lr_logz=LR_LOGZ):
    """Build the auxiliary GFlowNet with an independent backbone network.

    The auxiliary GFlowNet has the same architecture as the main model but
    entirely separate parameters — forward and backward policies share a
    trunk within the auxiliary model, but that trunk is distinct from the
    main model's trunk.  This matches the setup in Malek et al. (2025).
    """
    N = env.num

    module_PF_aux = MLP(N, env.n_actions, n_hidden_layers=4)
    module_PB_aux = MLP(N, env.n_actions - 1, trunk=module_PF_aux.trunk, n_hidden_layers=4)

    pf_estimator_aux = DiscretePolicyEstimator(module_PF_aux, env.n_actions)
    pb_estimator_aux = DiscretePolicyEstimator(module_PB_aux, env.n_actions, is_backward=True)

    gfn_aux = TBGFlowNet(pf_estimator_aux, pb_estimator_aux)
    sampler_aux = Sampler(pf_estimator_aux)

    non_logz_params = [v for k, v in dict(gfn_aux.named_parameters()).items() if k != "logZ"]

    optimizer_aux = Adam(non_logz_params, lr=lr_policy)

    logz_params = [dict(gfn_aux.named_parameters())["logZ"]]
    optimizer_aux.add_param_group({"params": logz_params, "lr": lr_logz})

    return gfn_aux, sampler_aux, optimizer_aux
