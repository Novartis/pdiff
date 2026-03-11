"""
Copyright (c) 2026 Novartis Biomedical Research Inc. Licensed under the MIT License. See LICENSE file in the project root.
"""

from skimage import transform
import numpy as np
from functools import partial
from torchvision import transforms


def get_resize_center_crop_numpy_transform(
    resize_size: int = 1080, crop_size: int = 512
) -> partial:
    """paramaterizes and returns a partial numpy-based image transform function.

    Args:
        resize_size (int, optional): square size to initialize the image to as a first step. Defaults to 1080.
        crop_size (int, optional): size of final center crop from resized image. Defaults to 512.

    Returns:
        partial: resize_center_crop function that can be applied directly, with parameters fixed.
    """
    return partial(
        _resize_center_crop_numpy_transform,
        resize_size=resize_size,
        crop_size=crop_size,
    )


def _resize_center_crop_numpy_transform(
    image: np.ndarray, resize_size: int = 1080, crop_size: int = 512
) -> np.ndarray:
    """resize then center crop an image as a HWC numpy ndarray

    Args:
        image (np.ndarray): HWC ndarray to transform.
        resize_size (int, optional): square dimension of initial resize. Defaults to 1080.
        crop_size (int, optional): size of final crop from resized image. Defaults to 512.

    Returns:
        np.ndarray: resized and center-cropped np.ndarray
    """
    image = np.array(image)
    image = transform.resize(image, (resize_size, resize_size), anti_aliasing=True)
    image = (image * 255).astype(np.uint8)
    starts = (np.array(image.shape[0:2]) - crop_size) // 2
    ends = starts + crop_size
    return image[starts[0] : ends[0], starts[1] : ends[1], :]


def get_resize_center_crop_pil_transforms(
    resize_size: int = 1080, crop_size: int = 512
) -> transforms.transforms:
    """paramaterizes and returns torchvision-based resize & center crop transforms that operate on PIL images.

    Args:
        resize_size (int, optional): square dimension of initial resize. Defaults to 1080.
        crop_size (int, optional): size of final crop from resized image. Defaults to 512.

    Returns:
        transforms.transforms: composed transforms to resize then center crop ((PIL) -> PIL)
    """
    return transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.ToPILImage(),
        ]
    )


def get_training_transforms(
    resize_size: int = 1080,
    crop_size: int = 512,
    center_crop: bool = False,
    random_flip: bool = True,
) -> transforms.transforms:
    """return torchvision transforms used in training, on pytorch tensors, with augmentations and normalization.

    Args:
        resize_size (int, optional): square dimension of initial resize. Defaults to 1080.
        crop_size (int, optional): size of final crop from resized image. Defaults to 512.
        center_crop (bool, optional): center crop instead of random crop. Defaults to False.
        random_flip (bool, optional): apply random flip augmentations. Defaults to True.

    Returns:
        transforms.transforms: torchvision transforms to apply operations ((PIL) -> Tensor)
    """
    return transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(crop_size)
            if center_crop
            else transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip()
            if random_flip
            else transforms.Lambda(lambda x: x),
            transforms.RandomVerticalFlip()
            if random_flip
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
