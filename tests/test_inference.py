"""
Copyright (c) 2026 Novartis Biomedical Research Inc. Licensed under the MIT License. See LICENSE file in the project root.
"""

from typing import Any, Dict

import numpy as np
from pdiff.model import pDiffModel
from pdiff.metadata import pDiffMetadata
import torch


def test_add_to_metadata(tmp_path):
    pDiffMetadata.initialize_dataframe(tmp_path / "add_test.pkl")
    test_pdiff_metadata = pDiffMetadata(tmp_path / "add_test.pkl")
    test_image = np.ones((128, 128, 3), dtype=np.uint8)
    test_pdiff_metadata.add_image_data(
        tmp_path / "extra_tmp", [test_image], "test_treatment", np.zeros(10)
    )
    assert np.array(test_pdiff_metadata.get_image(0)).shape == (128, 128, 3)


def test_inference_load(pdiff_model):
    assert pdiff_model.get_unet().device == torch.device("cpu")


def test_inference_predict(
    pdiff_model: pDiffModel, small_treatment_dict: Dict[str, Any], tmp_path
):
    test_filename = "predict_test.pkl"
    pdiff_model.predict(
        output_root_path=tmp_path,
        new_metadata_filename=test_filename,
        treatment_profile_dict=small_treatment_dict,
        gen_images_per_treatment=1,
        inference_steps=1,
        resolution=64,
    )
    prediction_pdiff_metadata = pDiffMetadata(tmp_path / test_filename)
    assert len(prediction_pdiff_metadata) == 1
    assert (
        prediction_pdiff_metadata.get_treatment_name(0)
        == list(small_treatment_dict.keys())[0]
    )
    assert np.array_equal(
        np.squeeze(prediction_pdiff_metadata.get_profile(0)),
        list(small_treatment_dict.values())[0],
    )
