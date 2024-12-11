"""Utilities for working with the dataset."""
import polars as pl

from pituitary_nfpa import locations
from pituitary_nfpa.enums import PredictionLabels


def prepare_dataset(
    target: PredictionLabels | str,
) -> list[dict[str, str | int]]:
    """Prepare a dataset for training.

    Parameters
    ----------
    label: str
        Which label to use as the prediction target. Either "progression",
        "recurrence", or "prediction-or-recurrence".

    Returns
    -------
    list[dict[str, str | int]]:
        List of dataset dictionaries in the format required to pass to a
        dataloader.

    """
    target = PredictionLabels(target)

    full_df = pl.read_excel(locations.dataset_workbook)
