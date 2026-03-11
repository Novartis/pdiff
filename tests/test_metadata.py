"""
Copyright (c) 2026 Novartis Biomedical Research Inc. Licensed under the MIT License. See LICENSE file in the project root.
"""

from pathlib import Path
import numpy as np
import pytest

from pdiff.metadata import pDiffMetadata
from typing import Dict, Any

pytestmark = pytest.mark.parametrize("metadata_instances", ["pdiff_relative_metadata"])


def test_initialize_dataframe(tmp_path, metadata_instances):
    init_path = Path(tmp_path / "init_test_metadata.pkl")
    pDiffMetadata.initialize_dataframe(init_path)
    assert init_path.is_file()
    init_pdiff_metadata = pDiffMetadata(init_path)
    assert len(init_pdiff_metadata) == 0
    assert len(init_pdiff_metadata.df.index.names) == 2


def test_datapath(relative_datapath: Path, metadata_instances):
    assert relative_datapath.is_file()


class TestpDiffMetadata:
    def test_pdiff_metadata_len(self, metadata_instances: pDiffMetadata, request):
        pdiff_metadata = request.getfixturevalue(metadata_instances)
        assert len(pdiff_metadata) > 0

    def test_pdiff_metadata_image(self, metadata_instances: pDiffMetadata, request):
        pdiff_metadata = request.getfixturevalue(metadata_instances)
        image = pdiff_metadata.get_image(0)
        assert np.array(image).shape == (2160, 2160, 3)

    def test_pdiff_metadata_profile(self, metadata_instances: pDiffMetadata, request):
        pdiff_metadata = request.getfixturevalue(metadata_instances)
        profile = pdiff_metadata.get_profile(0)
        assert profile.ndim == 2
        assert profile.shape[0] == 1

    def test_pdiff_metadata_treatment_name(
        self, metadata_instances: pDiffMetadata, request
    ):
        pdiff_metadata = request.getfixturevalue(metadata_instances)
        treatment_name = pdiff_metadata.get_treatment_name(0)
        assert treatment_name

    def test_get_treatment_dict(
        self,
        metadata_config: Dict[str, Any],
        metadata_instances: pDiffMetadata,
        sample_treatment,
        request,
    ):
        pdiff_metadata = request.getfixturevalue(metadata_instances)
        treatment_dict = pdiff_metadata.get_treatment_dict()
        assert len(treatment_dict) == 16
        assert isinstance(treatment_dict[sample_treatment], np.ndarray)
        assert treatment_dict[sample_treatment].ndim == 1
        assert (
            treatment_dict[sample_treatment].shape[0]
            == metadata_config["profile_length"]
        )

    def test_metadata_modify_image_paths_copy(
        self, metadata_instances: pDiffMetadata, tmp_path: Path, request
    ):
        pdiff_metadata = request.getfixturevalue(metadata_instances)
        new_root_dir = tmp_path / "test_root"
        pdiff_metadata.modify_image_paths(
            new_root_dir, path_levels_to_keep=4, do_copy=True
        )
        new_metadata = pDiffMetadata(new_root_dir / "metadata.pkl")
        assert np.array_equal(
            np.array(new_metadata.get_image(0)), np.array(pdiff_metadata.get_image(0))
        )


"""     def test_metadata_save_load(
        self,
        metadata_instances: pDiffMetadata, tmp_path: Path, request
    ):
        pdiff_metadata = request.getfixturevalue(metadata_instances)
        save_path = tmp_path / "test.pkl"
        save_path.parent.mkdir(exist_ok=True)
        pdiff_metadata.save_metadata(save_path)
        new_metadata = pDiffMetadata(save_path)
        assert np.array_equal(np.array(new_metadata.get_image(0)), np.array(pdiff_metadata.get_image(0))) """
