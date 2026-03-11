"""
Copyright (c) 2026 Novartis Biomedical Research Inc. Licensed under the MIT License. See LICENSE file in the project root.
"""

from pathlib import Path

import numpy as np
from pdiff.metadata import pDiffMetadata
from pdiff import image_transforms
from torch.utils.data import Dataset
from torch import tensor
from typing import Callable, Dict, Any
from typing_extensions import Self


class pDiffDataset(Dataset):
    """
    a pytorch mappable-style dataset for pdiff data. Each entry contains a treatment name, a profile, and an image
    """

    def __init__(
        self,
        pdiff_metadata: pDiffMetadata,
        profile_transform: Callable[[np.ndarray], Any] = tensor,
        image_transform: Callable[
            [Any], Any
        ] = image_transforms.get_training_transforms(),
    ):
        """initialize pDiffDataset

        Args:
            pdiff_metadata (pDiffMetadata): pDiffMetadata object to wrap as a pytorch dataset.
            profile_transform (Callable, optional): Transform to be applied to profile vectors. Seldom any need to set. Defaults to tensor.
            image_transform (Callable, optional): Transform to be applied to images from the pDiffMetadata. Typically want to resize and crop, either with torchvision for training or skimage methods for visualization. Defaults to image_transforms.get_training_transforms().
        """
        self.pdiff_metadata = pdiff_metadata
        self.profile_transform = profile_transform
        self.image_transform = image_transform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """dataset __getitem__ method by index

        Args:
            idx (int): index of element to access

        Returns:
            Dict[str, Any]: dictionary of relevant values for the idx-th entry (treatment_name, profile, and image)
        """
        profile = self.pdiff_metadata.get_profile(idx)
        if self.profile_transform is not None:
            profile = self.profile_transform(profile)
        image = self.pdiff_metadata.get_image(idx)
        if self.image_transform is not None:
            image = self.image_transform(image)
        treatment_name = self.pdiff_metadata.get_treatment_name(idx)
        return {"profile": profile, "treatment_name": treatment_name, "image": image}

    def __len__(self) -> int:
        """dataset __len__ method

        Returns:
            int: length of this dataset
        """
        return len(self.pdiff_metadata)

    def set_image_transform(self, image_transform: Callable[[Any], Any]) -> None:
        """set the image transform

        Args:
            image_transform (Callable[[np.ndarray], np.ndarray]): new image transform to apply to all images
        """
        self.image_transform = image_transform

    @staticmethod
    def from_file(pdiff_metadata_file_path: Path) -> Self:
        """static method to conveniently initialize a new pDiffDataset from a metadata file on disk directly

        Args:
            pdiff_metadata_file_path (Path): path to a file for pDiffMetata to load

        Returns:
            Self: newly initialized pDiffDataset instance
        """
        pdiff_metadata = pDiffMetadata(pdiff_metadata_file_path)
        return pDiffDataset(pdiff_metadata)
