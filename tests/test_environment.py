"""
Copyright (c) 2026 Novartis Biomedical Research Inc. Licensed under the MIT License. See LICENSE file in the project root.
"""

import torch


def test_torch_cuda():
    assert torch.version.cuda is not None
