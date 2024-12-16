"""Utilities for working with the dataset."""
import polars as pl

from pituitary_nfpa import locations
from pituitary_nfpa.constants import IMAGE_KEY, LABEL_KEY, MASK_KEY
from pituitary_nfpa.enums import PredictionLabels, DatasetSplit


def prepare_dataset(
    target: PredictionLabels | str,
    fold: int,
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
    """Prepare a dataset for training.

    Parameters
    ----------
    label: str
        Which label to use as the prediction target. Either "progression",
        "recurrence", or "prediction-or-recurrence".
    fold: int
        Which folds to use to generate train/test splits.

    Returns
    -------
    train_data: list[dict[str, str | int]]
        Training data. List of dataset dictionaries in the format required to
        pass to a dataloader.
    test_data: list[dict[str, str | int]]
        Testing data. List of dataset dictionaries in the format required to
        pass to a dataloader.

    """
    target = PredictionLabels(target)

    full_df = pl.read_parquet(locations.working_dataframe)

    # Assume missing data are negative
    full_df = full_df.filter(pl.col("include")).with_columns(
        pl.col("progression").fill_null(0),
        pl.col("recurrence").fill_null(0),
    )

    if target == PredictionLabels.PROGRESSION:
        target_col = "progression"
    elif target == PredictionLabels.RECURRENCE:
        target_col = "recurrence"
    elif target == PredictionLabels.PROGRESSION_OR_RECURRENCE:
        target_col = "label"
        full_df = full_df.with_columns(
            pl.max_horizontal("progression", "recurrence").alias(target_col)
        )

    def get_split(split_name: str) -> list[dict[str, str | int]]:
        split_df = full_df.filter(pl.col(f"fold_{fold}") == split_name)
        return [
            {
                IMAGE_KEY: image_path,
                MASK_KEY: seg_path,
                LABEL_KEY: label,
                "mrn": mrn,
                "accession": acc,
            }
            for (image_path, seg_path, label, mrn, acc) in split_df[
                ["image_path", "seg_path", target_col, "mrn", "accession"]
            ].iter_rows()
        ]

    return (
        get_split(DatasetSplit.TRAIN.value),
        get_split(DatasetSplit.TEST.value),
    )
