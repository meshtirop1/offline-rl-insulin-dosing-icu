"""
Network definition shared by policy learning (s5) and off-policy evaluation (s6).

Keeping it here means no pipeline stage has to import another stage, so the
numbered scripts stay independent and can be read in order.
"""
import torch.nn as nn

from config import CQL_HIDDEN


class QNet(nn.Module):
    """Feed-forward action-value network: state -> one value per discrete action."""

    def __init__(self, state_dim: int, n_actions: int, hidden: int = CQL_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)
