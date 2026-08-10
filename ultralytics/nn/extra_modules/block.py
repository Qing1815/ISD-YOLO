import torch.nn as nn

from ..modules.block import C3k2, C3k
from .metaformer import MetaFormerCGLUBlock, SepConv

__all__ = ["C3k2_ConvFormerCGLU"]


class C3k_ConvFormerCGLU(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(MetaFormerCGLUBlock(dim=c_, token_mixer=SepConv) for _ in range(n))
        )


class C3k2_ConvFormerCGLU(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_ConvFormerCGLU(self.c, self.c, n, shortcut, g)
            if c3k
            else MetaFormerCGLUBlock(dim=self.c, token_mixer=SepConv)
            for _ in range(n)
        )
