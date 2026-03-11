"""
Copyright (c) 2026 Novartis Biomedical Research Inc. Licensed under the MIT License. See LICENSE file in the project root.
"""

import numpy as np
from pdiff.image_transforms import get_resize_center_crop_numpy_transform


def test_get_resize_center_crop_numpy_transform():
    test_image = np.random.randint(
        0, np.iinfo(np.uint16).max, size=(2160, 2160, 3), dtype=np.uint16
    )
    transform = get_resize_center_crop_numpy_transform(resize_size=1080, crop_size=512)
    transformed_image = transform(test_image)
    assert transformed_image.dtype == np.uint8
    assert transformed_image.shape == (512, 512, 3)
