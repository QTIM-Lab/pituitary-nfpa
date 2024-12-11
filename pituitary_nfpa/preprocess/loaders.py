"""Loaders."""
import monai

from pituitary_nfpa.constants import IMAGE_KEY, MASK_KEY


def get_transforms(
    augmentation: bool,
) -> monai.transforms.Compose:
    """Get the transform pipeline.

    Parameters
    ----------
    augmentation: bool
        Include augmentation in the pipeline.

    Returns
    -------
    monai.transforms.Compose:
        Composed transform pipeline.

    """
    both_image_keys = [IMAGE_KEY, MASK_KEY]

    initial_transforms = [
        monai.transforms.LoadImaged(keys=both_image_keys),
    ]

    return monai.transforms.Compose(
        [
            *initial_transforms,
        ]
    )
