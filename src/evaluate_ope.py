"""
Off-policy evaluation (OPE) of the learned CQL policy against the clinician
(behavior) policy, on the patient-disjoint TEST split.

Why this file is the actual result
-----------------------------------
A learned Q-function's own Q-values are NOT evidence that its policy is good --
they can be inflated by bootstrapping/overestimation. To make a defensible claim
that policy pi_e is better (or no worse) than the clinicians, we estimate the
expected discounted return of pi_e from logged data using estimators with known
bias/variance properties, and put confidence intervals on them by bootstrapping
over patients (ICU stays). We report four quantities:

  1. Clinician value V(pi_b)   -- empirical discounted return in the logged data.
                                  This is the number the learned policy must beat.
  2. FQE                         -- Fitted Q Evaluation: model-based OPE, trains a
                                  separate Q^{pi_e} by Bellman backup under pi_e.
                                  Low variance, but biased if the Q model is wrong.
  3. WIS (per-decision)          -- self-normalized per-decision importance sampling.
                                  Model-free / unbiased-ish, but high variance;
                                  uses an estimated behavior policy pi_b(a|s).
  4. WDR                         -- Weighted Doubly Robust: combines FQE's Q model
                                  with an IS correction. Consistent if EITHER the
                                  Q model or the behavior model is right.

Target policy pi_e is a softmax over the frozen CQL Q-values with temperature
TAU (a deterministic argmax policy makes importance ratios degenerate, so we
evaluate a slightly-stochastic version of the learned policy). Behavior policy
pi_b(a|s) is estimated with a gradient-boosted classifier on the TRAIN split.

IMPORTANT INTERPRETATION LIMIT
------------------------------
All returns here are under the PROXY reward defined in build_episodes.py
(time-in-range shaping). Beating the clinician on this proxy is a methods result,
NOT a validated clinical-benefit claim. A clinical claim requires re-running this
against real outcomes (hypoglycemia incidence, mortality) once the PhysioNet
outcome linkage is in place, plus eICU external validation. Do not report the
numbers here as 'lives saved' or 'hypoglycemia reduced'.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier

from train_cql import QNet

RL_DIR = Path(r"C:\Users\mtiro\Downloads\glycemic\rl_insulin_dosing\data\rl")
MODELS = Path(r"C:\Users\mtiro\Downloads\glycemic\rl_insulin_dosing\models")
REPORT = Path(r"C:\Users\mtiro\Downloads\glycemic\rl_insulin_dosing\reports\ope.md")

GAMMA = 0.95
TAU = 0.5              # target-policy softmax temperature
RATIO_CLIP = 5.0       # clip per-step log importance ratio to [-5, 5]
PROB_FLOOR = 1e-3      # floor on behavior-policy probability
N_BOOTSTRAP = 200
FQE_EPOCHS = 40
SEED = 0


# ---------- data / trajectory helpers ----------
def load_split(name):
    d = np.load(RL_DIR / f"{name}.npz")
    return {k: d[k] for k in d.files}


def to_trajectories(split):
    """Group consecutive rows by icustay_id into list of index arrays."""
    ids = split["icustay_id"]
    trajs, start = [], 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or ids[i] != ids[start]:
            trajs.append(np.arange(start, i))
            start = i
    return trajs


# ---------- policies ----------
def target_policy_probs(q_net, states, tau=TAU):
    with torch.no_grad():
        q = q_net(torch.tensor(states, dtype=torch.float32))
        return F.softmax(q / tau, dim=1).numpy()


def fit_behavior_policy(train, n_actions):
    clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=300, learning_rate=0.1)
    clf.fit(train["state"], train["action"])
    return clf


def behavior_probs(clf, states, n_actions):
    p = clf.predict_proba(states)
    full = np.full((len(states), n_actions), PROB_FLOOR)
    for j, c in enumerate(clf.classes_):
        full[:, c] = np.maximum(p[:, j], PROB_FLOOR)
    return full / full.sum(axis=1, keepdims=True)


# ---------- estimator 1: clinician empirical value ----------
def clinician_value(split, trajs):
    vals = []
    for tr in trajs:
        r = split["reward"][tr]
        disc = GAMMA ** np.arange(len(r))
        vals.append(float((disc * r).sum()))
    return np.array(vals)  # per-trajectory returns


# ---------- estimator 2: FQE ----------
def train_fqe(train, q_net_target_probs_fn, n_actions, state_dim, epochs=FQE_EPOCHS):
    torch.manual_seed(SEED)
    fqe = QNet(state_dim, n_actions)
    fqe_target = QNet(state_dim, n_actions)
    fqe_target.load_state_dict(fqe.state_dict())
    opt = torch.optim.Adam(fqe.parameters(), lr=1e-3)

    s = torch.tensor(train["state"], dtype=torch.float32)
    a = torch.tensor(train["action"], dtype=torch.long)
    r = torch.tensor(train["reward"], dtype=torch.float32)
    s2 = torch.tensor(train["next_state"], dtype=torch.float32)
    done = torch.tensor(train["done"], dtype=torch.float32)
    pi_next = torch.tensor(q_net_target_probs_fn(train["next_state"]), dtype=torch.float32)

    n = s.shape[0]
    bs = 1024
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            with torch.no_grad():
                q_next = fqe_target(s2[idx])                      # (B, A)
                v_next = (pi_next[idx] * q_next).sum(dim=1)       # E_{a~pi_e}[Q(s',a)]
                target = r[idx] + GAMMA * (1 - done[idx]) * v_next
            q_sa = fqe(s[idx]).gather(1, a[idx].unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(q_sa, target)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for p, tp in zip(fqe.parameters(), fqe_target.parameters()):
                    tp.data.mul_(0.995).add_(0.005 * p.data)
    return fqe


def fqe_trajectory_values(fqe, split, trajs, pi_probs):
    """V(s0) = sum_a pi_e(a|s0) Q_fqe(s0,a) at each trajectory's initial state."""
    with torch.no_grad():
        q0 = fqe(torch.tensor(split["state"], dtype=torch.float32)).numpy()
    v = (pi_probs * q0).sum(axis=1)
    return np.array([float(v[tr[0]]) for tr in trajs])


# ---------- estimator 3 & 4: per-decision WIS and WDR ----------
def per_decision_weights(split, trajs, pi_e, pi_b):
    """Returns per-trajectory lists of cumulative (clipped) importance ratios."""
    a = split["action"]
    log_ratios = np.log(pi_e[np.arange(len(a)), a]) - np.log(pi_b[np.arange(len(a)), a])
    log_ratios = np.clip(log_ratios, -RATIO_CLIP, RATIO_CLIP)
    traj_cum = []
    for tr in trajs:
        traj_cum.append(np.cumsum(log_ratios[tr]))
    return traj_cum


def wis_value(split, trajs, traj_logw):
    """Self-normalized per-decision importance sampling.

    V = sum_t gamma^t * [ sum_i w_{i,t} r_{i,t} / sum_i w_{i,t} ], where the
    inner self-normalization is done per step in log-space for stability.
    """
    max_len = max(len(tr) for tr in trajs)
    per_step_logw = [[] for _ in range(max_len)]
    per_step_r = [[] for _ in range(max_len)]
    for tr, logw in zip(trajs, traj_logw):
        r = split["reward"][tr]
        for t in range(len(tr)):
            per_step_logw[t].append(logw[t])
            per_step_r[t].append(r[t])
    value = 0.0
    for t in range(max_len):
        lw = np.array(per_step_logw[t])
        rr = np.array(per_step_r[t])
        m = lw.max()
        w = np.exp(lw - m)
        wn = w / w.sum()
        value += (GAMMA ** t) * float((wn * rr).sum())
    return value


def wdr_value(split, trajs, traj_logw, fqe, pi_probs):
    """Weighted doubly-robust estimator."""
    with torch.no_grad():
        q_all = fqe(torch.tensor(split["state"], dtype=torch.float32)).numpy()
    a = split["action"]
    q_sa = q_all[np.arange(len(a)), a]
    v_s = (pi_probs * q_all).sum(axis=1)

    max_len = max(len(tr) for tr in trajs)
    per_step = [[] for _ in range(max_len)]   # (logw, r, q_sa, v_next)
    baseline0 = []
    for tr, logw in zip(trajs, traj_logw):
        r = split["reward"][tr]
        baseline0.append(v_s[tr[0]])
        for t in range(len(tr)):
            v_next = v_s[tr[t + 1]] if t + 1 < len(tr) else 0.0
            per_step[t].append((logw[t], r[t], q_sa[tr[t]], v_next))

    corr = 0.0
    for t in range(max_len):
        arr = per_step[t]
        lw = np.array([x[0] for x in arr])
        rr = np.array([x[1] for x in arr])
        qq = np.array([x[2] for x in arr])
        vn = np.array([x[3] for x in arr])
        m = lw.max()
        w = np.exp(lw - m)
        wn = w / w.sum()
        corr += (GAMMA ** t) * float((wn * (rr + GAMMA * vn - qq)).sum())
    return float(np.mean(baseline0)) + corr


# ---------- bootstrap ----------
def bootstrap_ci(estimator_fn, trajs, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    point = estimator_fn(trajs)
    boots = []
    n = len(trajs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(estimator_fn([trajs[i] for i in idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qnet", default=str(MODELS / "cql_qnet.pt"))
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--reward-key", default="reward",
                    help="which stored reward to evaluate under (e.g. 'reward' proxy "
                         "or 'mortality_reward' terminal survival outcome)")
    args = ap.parse_args()

    meta = json.loads((RL_DIR / "meta.json").read_text())
    n_actions = meta["n_actions"]
    state_dim = len(meta["feature_names"])

    train = load_split("train")
    test = load_split("test")
    if args.reward_key != "reward":
        if args.reward_key not in train:
            raise SystemExit(f"reward key '{args.reward_key}' not in dataset; re-run prepare_rl_dataset "
                             f"after add_outcomes.py")
        train["reward"] = train[args.reward_key]
        test["reward"] = test[args.reward_key]
    test_trajs = to_trajectories(test)

    q_net = QNet(state_dim, n_actions)
    q_net.load_state_dict(torch.load(args.qnet))
    q_net.eval()

    def tgt_probs(states):
        return target_policy_probs(q_net, states, tau=args.tau)

    print("fitting behavior policy...")
    beh_clf = fit_behavior_policy(train, n_actions)
    pi_b_test = behavior_probs(beh_clf, test["state"], n_actions)
    pi_e_test = tgt_probs(test["state"])

    print("training FQE...")
    fqe = train_fqe(train, tgt_probs, n_actions, state_dim)

    # per-trajectory arrays for estimators
    clin_returns = clinician_value(test, test_trajs)          # per-traj
    traj_logw = per_decision_weights(test, test_trajs, pi_e_test, pi_b_test)
    fqe_vals = fqe_trajectory_values(fqe, test, test_trajs, pi_e_test)

    # bootstrap over trajectory indices; build index-based estimators
    idx_map = {id(tr): k for k, tr in enumerate(test_trajs)}

    def clin_est(sample):
        return float(np.mean([clin_returns[idx_map[id(tr)]] for tr in sample]))

    def fqe_est(sample):
        return float(np.mean([fqe_vals[idx_map[id(tr)]] for tr in sample]))

    def wis_est(sample):
        logw = [traj_logw[idx_map[id(tr)]] for tr in sample]
        return wis_value(test, sample, logw)

    def wdr_est(sample):
        logw = [traj_logw[idx_map[id(tr)]] for tr in sample]
        return wdr_value(test, sample, logw, fqe, pi_e_test)

    print("bootstrapping estimators...")
    results = {}
    for name, fn in [("clinician_Vpi_b", clin_est), ("FQE", fqe_est),
                     ("WIS", wis_est), ("WDR", wdr_est)]:
        pt, lo, hi = bootstrap_ci(fn, test_trajs)
        results[name] = (pt, lo, hi)
        print(f"  {name:16s} = {pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    # action agreement + policy action mix for context
    with torch.no_grad():
        greedy = q_net(torch.tensor(test["state"], dtype=torch.float32)).argmax(1).numpy()
    agreement = float((greedy == test["action"]).mean())
    action_mix = np.bincount(greedy, minlength=n_actions) / len(greedy)

    lines = [
        "# Off-policy evaluation (test split, patient-disjoint)\n",
        f"gamma={GAMMA} target_softmax_tau={args.tau} ratio_clip=+/-{RATIO_CLIP} "
        f"prob_floor={PROB_FLOOR} n_bootstrap={N_BOOTSTRAP}\n",
        f"test trajectories (ICU stays): {len(test_trajs)}\n\n",
        "## Estimated discounted return under the PROXY reward\n",
        "| estimator | value | 95% CI |\n|---|---|---|\n",
    ]
    for name, (pt, lo, hi) in results.items():
        lines.append(f"| {name} | {pt:+.4f} | [{lo:+.4f}, {hi:+.4f}] |\n")

    clin_pt = results["clinician_Vpi_b"][0]
    fqe_pt, fqe_lo, fqe_hi = results["FQE"]
    wdr_pt, wdr_lo, wdr_hi = results["WDR"]
    beats = "ABOVE" if wdr_lo > clin_pt else ("OVERLAPS" if wdr_hi > clin_pt else "BELOW")
    lines.append(
        f"\n## Reading the result\n"
        f"Clinician (logged-policy) value under this proxy reward is {clin_pt:+.4f}. "
        f"The learned policy's WDR estimate is {wdr_pt:+.4f} [{wdr_lo:+.4f}, {wdr_hi:+.4f}], "
        f"whose 95% CI is **{beats}** the clinician point estimate. "
        f"FQE agrees at {fqe_pt:+.4f} [{fqe_lo:+.4f}, {fqe_hi:+.4f}].\n\n"
        f"Learned policy agreement with clinician actions (greedy): {agreement:.3f}\n\n"
        f"## Caveat (do not skip when writing up)\n"
        f"These returns are under the time-in-range PROXY reward, not a clinical outcome. "
        f"A positive gap here is a methods result about the proxy MDP -- it is NOT evidence of "
        f"reduced hypoglycemia or mortality. Those require the PhysioNet outcome linkage + eICU "
        f"external validation (see README). Also: WIS is high-variance at this horizon; when FQE "
        f"and WDR disagree with WIS, trust the doubly-robust WDR, and report all three so reviewers "
        f"can see the spread.\n"
    )

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("".join(lines), encoding="utf-8")
    print(f"\nwrote report to {args.report}")


if __name__ == "__main__":
    main()
