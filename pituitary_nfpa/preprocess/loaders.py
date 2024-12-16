"""Loaders."""
from typing import Sequence

import numpy as np

from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import (
    Orientationd,
    RandAffined,
    RandFlipd,
    ResampleToMatchd,
    Spacingd,
)
from monai.transforms.croppad.dictionary import CenterSpatialCropd
from monai.transforms.utility.dictionary import (
    DeleteItemsd,
    EnsureChannelFirstd,
    ToTensord,
    ToDeviced,
)
from monai.transforms.intensity.dictionary import (
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandGaussianNoised,
)
import torch
from qtim_transforms import ForegroundCenteredCropd

from pituitary_nfpa.constants import IMAGE_KEY, MASK_KEY, LABEL_KEY


def get_transforms(
    augmentation: bool = False,
    crop_size: Sequence[int] = (128, 128, 6),
    use_gpu: bool = False,
) -> Compose:
    """Get the transform pipeline.

    Parameters
    ----------
    augmentation: bool, optional
        Include augmentation in the pipeline.
    crop_size: Sequence[int], optional
        Size of the cropped region around the segmentation.
    use_gpu: bool, optional
        Whether to use the GPU for preprocessing transforms.

    Returns
    -------
    monai.transforms.Compose:
        Composed transform pipeline.

    """
    both_image_keys = [IMAGE_KEY, MASK_KEY]
    max_rotation_deg = 15.0
    max_rotation_rad = np.deg2rad(max_rotation_deg)
    padding_factor = 1 + 2.2 * np.cos(max_rotation_rad)
    rotate_range = [
        (-max_rotation_rad, max_rotation_rad),
        (-max_rotation_rad, max_rotation_rad),
        (0, 0),
    ]

    transpose_xy = 10
    transpose_z = 3
    translate_range = [
        (-transpose_xy, transpose_xy),
        (-transpose_xy, transpose_xy),
        (-transpose_z, transpose_z),
    ]

    outer_crop_size = (
        int(np.ceil(crop_size[0] * padding_factor)) + 2 * transpose_xy,
        int(np.ceil(crop_size[1] * padding_factor)) + 2 * transpose_xy,
        2 * transpose_z,
    )

    device = "cuda:0" if use_gpu else "cpu"

    initial_transforms = [
        ToTensord(keys=[LABEL_KEY], dtype=torch.float32),
        LoadImaged(keys=both_image_keys),
        *([ToDeviced(keys=both_image_keys, device=device)] if use_gpu else []),
        EnsureChannelFirstd(keys=both_image_keys),
        Orientationd(
            axcodes="ILA",
            keys=both_image_keys,
            lazy=True,
        ),
        Spacingd(
            pixdim=[0.3, 0.3, 2.0],
            keys=[IMAGE_KEY],
            mode=["bilinear"],
            lazy=True,
        ),
        ResampleToMatchd(
            key_dst=IMAGE_KEY,
            keys=[MASK_KEY],
            lazy=True,
            padding_mode="zeros",
            mode="nearest",
        ),
        ForegroundCenteredCropd(
            source_key=MASK_KEY,
            keys=both_image_keys,
            roi_size=outer_crop_size,
            lazy=True,
        ),
        DeleteItemsd(keys=[MASK_KEY]),
        NormalizeIntensityd(
            keys=[IMAGE_KEY],
        ),
    ]

    spatial_aug_transforms = []
    intensity_aug_transforms = []
    if augmentation:
        spatial_aug_transforms = [
            RandAffined(
                keys=[IMAGE_KEY],
                prob=0.9,
                rotate_range=rotate_range,
                translate_range=translate_range,
                scale_range=(-0.15, 0.15),
                mode="bilinear",
                lazy=True,
            ),
            RandFlipd(
                keys=[IMAGE_KEY],
                prob=0.5,
                spatial_axis=[1],
            ),
        ]

    crop_transform = CenterSpatialCropd(
        keys=[IMAGE_KEY],
        roi_size=crop_size,
    )

    if augmentation:
        intensity_aug_transforms = [
            RandScaleIntensityd(
                keys=[IMAGE_KEY],
                prob=0.2,
                factors=(-0.1, 0.1),
            ),
            RandShiftIntensityd(
                keys=[IMAGE_KEY],
                prob=0.2,
                offsets=(-0.1, 0.1),
            ),
            RandGaussianSmoothd(
                keys=[IMAGE_KEY],
                prob=0.2,
                sigma_x=(0, 1.5),
                sigma_y=(0, 1.5),
            ),
            RandGaussianSharpend(
                keys=[IMAGE_KEY],
                prob=0.2,
            ),
            RandGaussianNoised(
                keys=[IMAGE_KEY],
                prob=0.2,
                std=0.1,
            ),
            RandAdjustContrastd(
                keys=[IMAGE_KEY],
                prob=0.3,
            ),
        ]

    return Compose(
        [
            *initial_transforms,
            *spatial_aug_transforms,
            crop_transform,
            *intensity_aug_transforms,
        ],
        lazy=True,
    )
