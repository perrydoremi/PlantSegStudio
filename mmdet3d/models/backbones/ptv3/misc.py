"""
General Utils for Models

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.

VENDORED from Pointcept (pointcept/models/utils/misc.py).
Modification: kept only the offset/batch helpers needed by PTv3; dropped the
unused `off_diagonal`.
"""

import torch


@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.no_grad()
def bincount2offset(bincount):
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()
