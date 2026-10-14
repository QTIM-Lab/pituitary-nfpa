"""Loaders."""
from typing import Sequence

import numpy as np

from monai.transforms import RandomizableTransform, Transform
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
    Identityd,
    SqueezeDimd,
    ToDeviced,
    ToTensord,
)
from monai.transforms.intensity.dictionary import (
    MaskIntensityd,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
import torch
from qtim_transforms import ForegroundCenteredCropd

from pituitary_nfpa.constants import IMAGE_KEY, MASK_KEY, LABEL_KEY


DEFAULT_CROP_SIZE = (128, 128, 6)


class ForceCache(Identityd, RandomizableTransform):
    """Hack transform to force caching while having no other effect."""

    pass


def get_transforms(
    augmentation: bool = False,
    crop_size: Sequence[int] | None = None,
    use_gpu: bool = False,
    use_cache: bool = False,
    single_slice: bool = False,
    mask_foreground: bool = False,
) -> Compose:
    """Get the transform pipeline.

    Parameters
    ----------
    augmentation: bool, optional
        Include augmentation in the pipeline.
    crop_size: Sequence[int] | None, optional
        Size of the cropped region around the segmentation. If None,
        a default will be applied based on the value of single_slice.
    use_gpu: bool, optional
        Whether to use the GPU for preprocessing transforms.
    single_slice: bool, optional
        Return a 2D tensor with a single slice.
    mask_foreground: bool, optional
        Mask out (set to black) anything outside the segmentation mask.

    Returns
    -------
    monai.transforms.Compose:
        Composed transform pipeline.

    """
    both_image_keys = [IMAGE_KEY, MASK_KEY]
    output_keys = [IMAGE_KEY, LABEL_KEY]
    max_rotation_deg = 15.0
    max_rotation_rad = np.deg2rad(max_rotation_deg)
    padding_factor = 1 + 2.2 * np.cos(max_rotation_rad)

    if crop_size is None:
        crop_size = (
            DEFAULT_CROP_SIZE[:2] + (1, )
            if single_slice
            else DEFAULT_CROP_SIZE
        )

    transpose_xy = 10
    transpose_z = 3

    outer_crop_y = int(np.ceil(crop_size[0] * padding_factor)) + 2 * transpose_xy
    outer_crop_x = int(np.ceil(crop_size[1] * padding_factor)) + 2 * transpose_xy

    rotate_range = [
        (-max_rotation_rad, max_rotation_rad),
        (-max_rotation_rad, max_rotation_rad),
    ]
    resample_spacing = (0.3, 0.3, 0.2)

    if single_slice:
        translate_range = [
            (-transpose_xy, transpose_xy),
            (-transpose_xy, transpose_xy),
            (0, 0)
        ]
        outer_crop_size = (outer_crop_y, outer_crop_x, 1)
    else:
        translate_range = [
            (-transpose_xy, transpose_xy),
            (-transpose_xy, transpose_xy),
            (-transpose_z, transpose_z),
        ]
        outer_crop_size = (outer_crop_y, outer_crop_x, 2 * transpose_z + 1)

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
            pixdim=resample_spacing,
            keys=[MASK_KEY],
            mode=["bilinear"],
            lazy=True,
        ),
        ResampleToMatchd(
            key_dst=MASK_KEY,
            keys=[IMAGE_KEY],
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
        NormalizeIntensityd(
            keys=[IMAGE_KEY],
        ),
        *(
            [MaskIntensityd(keys=[IMAGE_KEY], mask_key=MASK_KEY)]
            if mask_foreground
            else []
        ),
        DeleteItemsd(keys=[MASK_KEY]),
    ]

    cache_transforms: list[Transform] = []
    if use_cache and augmentation:
        if use_gpu:
            # Force caching on the CPU
            cache_transforms.append(ToDeviced(keys=output_keys, device="cpu"))
        cache_transforms.append(ForceCache(keys=output_keys))
        if use_gpu:
            # Return to gpu after cache
            cache_transforms.append(ToDeviced(keys=output_keys, device=device))

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

    final_transforms = [
        *(
            [
                SqueezeDimd(
                    keys=[IMAGE_KEY],
                    dim=-1,
                    update_meta=False
                )
            ]
            if single_slice else []
        ),
        # Allows CUDA tensors to be shared between threads
        ToTensord(output_keys, track_meta=False),
    ]

    return Compose(
        [
            *initial_transforms,
            *cache_transforms,
            *spatial_aug_transforms,
            crop_transform,
            *intensity_aug_transforms,
            *final_transforms,
        ],
        lazy=True,
    )
