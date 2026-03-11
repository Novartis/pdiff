"""
Copyright (c) 2026 Novartis Biomedical Research Inc. Licensed under the MIT License. See LICENSE file in the project root.
"""

from pdiff.dataset import pDiffDataset


def test_dataset_len(pdiff_dataset: pDiffDataset):
    assert len(pdiff_dataset)


def test_dataset_getitem(pdiff_dataset: pDiffDataset):
    sample = pdiff_dataset[0]
    assert "profile" in sample.keys()
    assert "treatment_name" in sample.keys()
    assert "image" in sample.keys()


def test_dataset_getitem_image(pdiff_dataset: pDiffDataset):
    sample = pdiff_dataset[0]
    assert sample["image"].ndim == 3
    assert sample["profile"].ndim == 2
