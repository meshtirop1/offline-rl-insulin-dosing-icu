"""
Discrete Conservative Q-Learning (CQL) for the 16-action insulin dosing
policy defined in prepare_rl_dataset.py.

Loss (standard discrete CQL, Kumar et al. 2020):
    L = TD_loss(Q, target)  +  alpha * E_s[ logsumexp_a Q(s,a) - Q(s, a_data) ]
The second term pushes down Q for actions not seen in the data at state s,
and pushes up Q for the action the clinician actually took -- this is what
makes the learned policy conservative w.r.t. distribution shift, which
matters a lot here given several actions have <1000 supporting transitions.

TD target uses Double-DQN style decoupling (online net picks the argmax
action, target net evaluates it) to reduce overestimation, with a target
network updated by Polyak averaging.

This is a from-scratch implementation (not d3rlpy) so every term is legible
and citable in a methods section.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RL_DIR = Path(r"C:\Users\mtiro\Downloads\glycemic\rl_insulin_dosing\data\rl")
OUT_DIR = Path(r"C:\Users\mtiro\Downloads\glycemic\rl_insulin_dosing\models")
REPORT = Path(r"C:\Users\mtiro\Downloads\glycemic\rl_insulin_dosing\reports\cql.md")

GAMMA = 0.95          # per 4h bin; ~1 day horizon dominates the discount
ALPHA_CQL = 1.0        # conservative penalty weight (override with --alpha)
LR = 1e-3
BATCH_SIZE = 512
EPOCHS = 60
TARGET_TAU = 0.005     # polyak averaging rate
HIDDEN = 128
SEED = 0


class QNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def load_split(name):
    d = np.load(RL_DIR / f"{name}.npz")
    return {
        "state": torch.tensor(d["state"], dtype=torch.float32),
        "action": torch.tensor(d["action"], dtype=torch.long),
        "reward": torch.tensor(d["reward"], dtype=torch.float32),
        "next_state": torch.tensor(d["next_state"], dtype=torch.float32),
        "done": torch.tensor(d["done"], dtype=torch.float32),
    }


def cql_loss_fn(q_net, target_net, batch, n_actions, alpha_cql):
    s, a, r, s2, done = batch["state"], batch["action"], batch["reward"], batch["next_state"], batch["done"]

    q_all = q_net(s)                      # (B, A)
    q_sa = q_all.gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_online = q_net(s2)
        next_action = next_q_online.argmax(dim=1, keepdim=True)
        next_q_target = target_net(s2).gather(1, next_action).squeeze(1)
        td_target = r + GAMMA * (1 - done) * next_q_target

    td_loss = F.mse_loss(q_sa, td_target)

    logsumexp_q = torch.logsumexp(q_all, dim=1)
    cql_term = (logsumexp_q - q_sa).mean()

    loss = td_loss + alpha_cql * cql_term
    return loss, td_loss.item(), cql_term.item(), q_sa.mean().item()


def polyak_update(online, target, tau):
    with torch.no_grad():
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)


def evaluate(q_net, batch, n_actions):
    with torch.no_grad():
        q_all = q_net(batch["state"])
        greedy = q_all.argmax(dim=1)
        agreement = (greedy == batch["action"]).float().mean().item()
        mean_q_at_data_action = q_all.gather(1, batch["action"].unsqueeze(1)).mean().item()
        mean_q_at_greedy = q_all.gather(1, greedy.unsqueeze(1)).mean().item()
        action_dist = torch.bincount(greedy, minlength=n_actions).float()
        action_dist = (action_dist / action_dist.sum()).tolist()
    return agreement, mean_q_at_data_action, mean_q_at_greedy, action_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=ALPHA_CQL, help="CQL conservative penalty weight")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default=str(OUT_DIR / "cql_qnet.pt"))
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args()
    alpha_cql = args.alpha

    torch.manual_seed(SEED)
    meta = json.loads((RL_DIR / "meta.json").read_text())
    n_actions = meta["n_actions"]
    state_dim = len(meta["feature_names"])

    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    q_net = QNet(state_dim, n_actions)
    target_net = QNet(state_dim, n_actions)
    target_net.load_state_dict(q_net.state_dict())
    opt = torch.optim.Adam(q_net.parameters(), lr=LR)

    n = train["state"].shape[0]
    lines = ["# Discrete CQL training\n", f"gamma={GAMMA} alpha_cql={alpha_cql} lr={LR} "
             f"batch_size={BATCH_SIZE} epochs={args.epochs} n_actions={n_actions}\n"]

    prev_val_q = None
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        ep_loss, ep_td, ep_cql = 0.0, 0.0, 0.0
        n_batches = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            batch = {k: v[idx] for k, v in train.items()}
            loss, td, cql, _ = cql_loss_fn(q_net, target_net, batch, n_actions, alpha_cql)
            opt.zero_grad()
            loss.backward()
            opt.step()
            polyak_update(q_net, target_net, TARGET_TAU)
            ep_loss += loss.item(); ep_td += td; ep_cql += cql; n_batches += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            val_agree, val_q_data, val_q_greedy, _ = evaluate(q_net, val, n_actions)
            drift = "" if prev_val_q is None else f"  dQ={val_q_greedy - prev_val_q:+.4f}"
            prev_val_q = val_q_greedy
            msg = (f"epoch {epoch:3d}  loss={ep_loss/n_batches:.4f}  td={ep_td/n_batches:.4f}  "
                   f"cql={ep_cql/n_batches:.4f}  val_action_agreement={val_agree:.3f}  "
                   f"val_Q(data_action)={val_q_data:.3f}  val_Q(greedy)={val_q_greedy:.3f}{drift}")
            print(msg)
            lines.append(msg + "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(q_net.state_dict(), Path(args.out))

    test_agree, test_q_data, test_q_greedy, action_dist = evaluate(q_net, test, n_actions)
    action_meta = meta["action_meta"]
    dist_lines = "\n".join(
        f"  action {m['action_id']:2d} ({m['route']}/bin{m['dose_bin']}): "
        f"{action_dist[m['action_id']]*100:5.1f}%"
        for m in action_meta
    )

    summary = (
        f"\n## Test-set results\n"
        f"action agreement with clinician (greedy policy == logged action): {test_agree:.3f}\n"
        f"mean Q at clinician's action: {test_q_data:.3f}\n"
        f"mean Q at learned greedy action: {test_q_greedy:.3f}\n"
        f"(higher Q at greedy than at data action is expected -- that's the policy improvement "
        f"signal; the open question, deferred to formal OPE, is whether that Q gap reflects a "
        f"real improvement or overestimation bias)\n\n"
        f"## Learned greedy policy's action distribution on test states\n{dist_lines}\n\n"
        f"## Caveat\n"
        f"This is training-time diagnostics only (TD loss convergence, CQL conservatism, "
        f"action agreement). It is NOT a validated estimate of clinical benefit -- that requires "
        f"formal off-policy evaluation (FQE / weighted importance sampling / doubly robust) against "
        f"a real outcome (hypoglycemia incidence, mortality), which needs the PhysioNet "
        f"ADMISSIONS/PATIENTS outcome linkage described in the project README. Do not treat "
        f"'Q(greedy) > Q(data)' above as evidence the learned policy is safer.\n"
    )
    print(summary)
    lines.append(summary)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote model to {args.out}")
    print(f"wrote report to {report_path}")


if __name__ == "__main__":
    main()
