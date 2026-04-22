import pytest

import torch

from conf.paths import ROOT


@pytest.fixture
def tensors():
    tensors = [
        # 1d
        torch.randn(1),
        torch.zeros(3),
        torch.randn(5),
        torch.ones(100),
        # 2d
        torch.randn(3, 3),
        torch.randn(8, 1),
        torch.randn(1, 9),
        torch.randn(7, 15),
        # 3d
        torch.randn(1, 1, 3),
        torch.randn(3, 1, 1),
        torch.randn(1, 3, 1),
        torch.randn(4, 5, 8),
    ]
    return tensors


@pytest.fixture
def path_exec():
    return ROOT / "main.py"
