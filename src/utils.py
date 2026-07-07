import numpy as np

from src.config import K, STEP, MAXWEIGHT


def sampleportfolio(tickers):
    N = len(tickers)
    S = np.round(1 / STEP)
    maxstep = np.round(MAXWEIGHT / STEP)

    k = np.random.randint(np.ceil(S / maxstep), K + 1)

    ids = np.random.choice(N, k, False)

    q = np.zeros(N)

    q_val = np.random.multinomial(S, np.ones(k) / k)
    q_val = np.minimum(q_val, maxstep)
    s = int(q_val.sum())

    while s < S:
        cap = np.where(q_val < maxstep)[0]
        if cap.size == 0:
            return sampleportfolio(tickers)
        j = np.random.choice(cap)
        q_val[j] += 1
        s += 1

    while s > S:
        pos = np.where(q_val > 0)[0]
        j = np.random.choice(pos)
        q_val[j] -= 1
        s -= 1

    q[ids] = q_val

    w = q * STEP

    return w


def sharpe(r):
    mu = r.mean()
    sd = r.std(ddof=1)
    return float(np.sqrt(252) * mu / (sd))


def mdd(r):
    wealth = np.exp(np.cumsum(r))
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    return float(-dd.min())


def evaluate_portfolio(w, data):
    active = np.nonzero(w)

    val = data[:, active]

    r = np.matmul(val, w[active])
    return {
        "sharpe": sharpe(r),
        "mdd":    mdd(r),
    }


def paretorank(sharpes, mdds):
    N = len(sharpes)
    ranks = np.full(N, -1)
    remaining = set(range(N))
    rank = 0
    while remaining:
        front = []
        for i in list(remaining):
            dominated = False
            for j in remaining:
                if j == i:
                    continue
                weakdom = (sharpes[j] >= sharpes[i]) and (mdds[j] <= mdds[i])
                strong = (sharpes[j] > sharpes[i]) or (mdds[j] < mdds[i])
                if weakdom and strong:
                    dominated = True
                    break
            if not dominated:
                front.append(i)

        for i in front:
            ranks[i] = rank
        remaining -= set(front)
        rank += 1
    return ranks


def reward(ranks):
    num_fronts = ranks.max() + 1
    return np.exp(5 * (num_fronts - ranks) / num_fronts)
