"""
Copyright (c) 2026 Novartis Biomedical Research Inc. Licensed under the MIT License. See LICENSE file in the project root.
"""

from pathlib import Path
from typing import Any, Tuple, Union, Dict
from cellpose import models
import numpy as np


default_cellpose_args_dict = {
    "flow_threshold": 0.8,
    "cellprob_threshold": 0,
    "diameter": 60,
    "channels": [0, 0],
}


def init_cellpose(
    model_type="cyto2", model_path: Union[Path, str] = None, device=None
) -> models.CellposeModel:
    """initialize a CellposeModel on provided device.

    Args:
        model_type (str, optional): model type string to pass to cellpose. Defaults to "cyto2".
        model_path (_type_, optional): model path for custom model, overrides model_type. Defaults to None.
        device (_type_, optional): pytorch device to. Defaults to None.

    Returns:
        CellposeModel: initialized model on the provided device
    """
    device, gpu = models.assign_device(True, True, device)
    cp_model = None
    if model_path is not None:
        cp_model = models.CellposeModel(
            gpu=gpu, device=device, pretrained_model=model_path
        )
    else:
        cp_model = models.CellposeModel(gpu=gpu, device=device, model_type=model_type)
    return cp_model


def run_cellpose(
    images: Any, cp_model: models.CellposeModel, cellpose_args_dict: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate provided cellpose model with arguments on the provided images, and return masks and style vectors.

    Args:
        images (Any): images to run cellpose against
        cp_model (models.CellposeModel): CellposeModel to apply
        cellpose_args_dict (dict[str, Any]): contains arguments for cellpose eval()

    Returns:
        tuple[np.ndarray, np.ndarray]: image masks (integer 1-channel) and style vectors (1D array)
    """
    masks, flows, styles = cp_model.eval(
        images,
        invert=False,
        compute_masks=True,
        augment=True,
        flow_threshold=float(cellpose_args_dict["flow_threshold"]),
        cellprob_threshold=float(cellpose_args_dict["cellprob_threshold"]),
        diameter=int(cellpose_args_dict["diameter"]),
        channels=cellpose_args_dict["channels"],
        resample=True,
    )
    return masks, styles
