"""
Copyright (c) 2026 Novartis Biomedical Research Inc. Licensed under the MIT License. See LICENSE file in the project root.
"""

from pathlib import Path
from typing import Any, Dict
import pytest

from pdiff.dataset import pDiffDataset
from pdiff.metadata import pDiffMetadata
from pdiff.model import pDiffModel

# @pytest.fixture
# def datapath():
#     test_path = Path(__file__).resolve().parent
#     datapath = test_path / "test_data/images/prepared_metadata.pkl"
#     return datapath


@pytest.fixture
def relative_datapath():
    test_path = Path(__file__).resolve().parent
    datapath = test_path / "test_data/images/relative_metadata.pkl"
    return datapath


@pytest.fixture
def oneline_datapath():
    test_path = Path(__file__).resolve().parent
    datapath = test_path / "test_data/images/oneline_metadata.pkl"
    return datapath


@pytest.fixture
def metadata_config():
    config_dict = {
        "image_channels": ["Location_ch2", "Location_ch3", "Location_ch1"],
        "profile": "morgan_fingerprint",
        "profile_length": 2048,
    }
    return config_dict


@pytest.fixture
def sample_treatment() -> str:
    return "chlorambucil"


@pytest.fixture
def cellpose_args_dict() -> Dict[str, Any]:
    cellpose_args_dict = {
        "flow_threshold": 0.8,
        "cellprob_threshold": 0,
        "diameter": 60,
        "channels": [0, 0],
    }
    return cellpose_args_dict


@pytest.fixture
def pdiff_oneline_metadata(oneline_datapath) -> pDiffMetadata:
    my_pdiffmetadata = pDiffMetadata(oneline_datapath)
    return my_pdiffmetadata


# @pytest.fixture
# def pdiff_metadata(datapath) -> pDiffMetadata:
#     my_pdiffmetadata = pDiffMetadata(datapath)
#     return my_pdiffmetadata


@pytest.fixture
def pdiff_relative_metadata(relative_datapath) -> pDiffMetadata:
    my_pdiffmetadata = pDiffMetadata(relative_datapath)
    return my_pdiffmetadata


@pytest.fixture
def small_treatment_dict(sample_treatment, pdiff_relative_metadata):
    treatment_dict = pdiff_relative_metadata.get_treatment_dict()
    profile_value = treatment_dict[sample_treatment]
    return {sample_treatment: profile_value}


@pytest.fixture
def pdiff_dataset(pdiff_relative_metadata: pDiffMetadata):
    return pDiffDataset(pdiff_relative_metadata)


@pytest.fixture
def pdiff_model():
    model_path = Path(__file__).resolve().parent / "test_data/sample_pdiff_model"
    return pDiffModel(model_path=model_path, from_scratch=True)
