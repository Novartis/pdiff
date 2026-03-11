"""
Copyright (c) 2026 Novartis Biomedical Research Inc. Licensed under the MIT License. See LICENSE file in the project root.
"""

from pdiff.image_transforms import get_resize_center_crop_numpy_transform
from pdiff.metadata import pDiffMetadata


def test_cellpose_analysis(pdiff_oneline_metadata, cellpose_args_dict, tmp_path):
    pdiff_oneline_metadata.save(tmp_path / "cellpose_test.pkl")
    image_transform = get_resize_center_crop_numpy_transform(
        resize_size=540, crop_size=256
    )
    pdiff_oneline_metadata.apply_cellpose(
        tmp_path / "cellpose_mask_test",
        cellpose_args_dict,
        image_transform,
    )
    assert pDiffMetadata.mask_string in pdiff_oneline_metadata.df.columns
    assert (
        pDiffMetadata.extracted_image_fingerprint_string
        in pdiff_oneline_metadata.df.columns
    )
    mask = pdiff_oneline_metadata.get_mask(0)
    assert mask.ndim == 2
