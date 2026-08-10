import torch.nn as nn

from ..modules.block import C2PSA, PSABlock
from .attention import BiLevelRoutingAttention_nchw

__all__ = ["C2BRA"]


class BRABlock(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.attn = BiLevelRoutingAttention_nchw(dim=c, num_heads=num_heads)


class C2BRA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(
            *(BRABlock(self.c, attn_ratio=0.5, num_heads=max(1, self.c // 64)) for _ in range(n))
        )
