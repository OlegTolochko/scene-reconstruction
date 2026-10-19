import torch.nn as nn

from ..basic import VarLenTensor
from ..linear import SparseLinear
from ..nonlinearity import SparseGELU


class SparseFeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, int(channels * mlp_ratio)),
            SparseGELU(approximate="tanh"),
            SparseLinear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: VarLenTensor) -> VarLenTensor:
        return self.mlp(x)
