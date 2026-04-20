"""Run interpretability of TabPFN model."""
from enum import Enum
from pathlib import Path

import click
import numpy as np
import polars as pl
from pycrumbs import tracked
import tabpfn

from pituitary_nfpa import locations
from pituitary_nfpa.preprocess.features import get_clinical_features
from pituitary_nfpa.constants import (
    VAL_RATE,
    N_FOLDS,
)

import shapiq
import matplotlib.pyplot as plt


class RadiomicFeatureGroups(Enum):
    """Enumeration for groups of radiomic features that can be used."""

    SHAPE = "SHAPE"
    FIRST_ORDER = "FIRST_ORDER"  # includes shape
    ALL = "ALL"


@click.command()
@click.argument(
    "output-dir",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
)
@click.option(
    "--radiomic-groups",
    "-g",
    type=click.Choice([v.value for v in RadiomicFeatureGroups]),
    default="ALL",
    help="Which groups of radiomic features to include.",
)
@click.option(
    "--use-clinical/--no-use-clinical",
    "-c/-C",
    default=True,
    help="Include clinical features",
)
@click.option(
    "--use-radiomics/--no-use-radiomics",
    "-r/-R",
    default=True,
    help="Include radiomic features",
)
@tracked(
    directory_parameter="output_dir",
)
def interpret_tabpfn(
    output_dir: Path,
    use_clinical: bool,
    use_radiomics: bool,
    radiomic_groups: str = "ALL",
) -> None:
    """Train a tabular model.

    Inputs are some combination of clinical, radiomic, and CNN predictions.

    This will create a new directory in the checkpoints directory containing
    model weights and any other files associated with model training.

    MODEL_NAME is a free text string that names the model. The output will be
    placed in a directory with this name.

    """
    if not output_dir.exists():
        output_dir.mkdir()

    features_df = pl.read_parquet(locations.radiomic_features_dataframe)
    meta_df = pl.read_parquet(locations.working_dataframe)

    joined_df = features_df.join(
        meta_df,
        how="left",
        on=("mrn", "accession", "case_number"),
        validate="1:1",
    )

    if not use_radiomics and not use_clinical:
        raise ValueError("Must use either clinical or radiomic features.")

    for fold in range(N_FOLDS):

        train_df = joined_df.filter(pl.col(f"fold_{fold}") == "TRAIN")
        test_df = joined_df.filter(pl.col(f"fold_{fold}") == "TEST")

        val_cutoff = int((1.0 - VAL_RATE) * len(train_df))
        val_df = train_df[val_cutoff:]
        train_df = train_df[:val_cutoff]

        train_features_list = []
        val_features_list = []
        test_features_list = []
        if use_radiomics:
            if radiomic_groups == "SHAPE":
                feature_cols = [
                    c for c in features_df.columns if c.startswith("original_shape")
                ]
            elif radiomic_groups == "FIRST_ORDER":
                feature_cols = [
                    c
                    for c in features_df.columns
                    if c.startswith("original_shape")
                    or c.startswith("original_first_order")
                ]
            elif radiomic_groups == "ALL":
                feature_cols = [
                    c for c in features_df.columns if c.startswith("original")
                ]
            else:
                raise ValueError("Unrecognized radiomic group ")

            train_features_list.append(train_df[feature_cols].to_numpy())
            val_features_list.append(val_df[feature_cols].to_numpy())
            test_features_list.append(test_df[feature_cols].to_numpy())
        if use_clinical:
            train_features_list.append(
                get_clinical_features(train_df, impute_missing=False)
            )
            val_features_list.append(
                get_clinical_features(val_df, impute_missing=False)
            )
            test_features_list.append(
                get_clinical_features(test_df, impute_missing=False)
            )

        train_features = np.column_stack(train_features_list)
        # val_features = np.column_stack(val_features_list)
        test_features = np.column_stack(test_features_list)

        target_col = "progression_or_recurrence"
        train_labels = np.squeeze(train_df[[target_col]].to_numpy(), 1)
        test_labels = np.squeeze(test_df[[target_col]].to_numpy(), 1)
        # val_labels = np.squeeze(val_df[[target_col]].to_numpy(), 1)

        estimator = tabpfn.TabPFNClassifier()
        estimator.fit(train_features, train_labels)

        explainer = shapiq.TabPFNExplainer(
            model=estimator,
            data=test_features,
            labels=test_labels,
            max_order=1,
            verbose=True,
        )

        print("Explaining")
        explanations = explainer.explain_X(test_features, budget=256, n_jobs=32)
        shapiq.beeswarm_plot(explanations, test_features, show=False)
        plt.savefig(output_dir / f"beeswarm_fold{fold}.png")
        plt.close()
        shapiq.bar_plot(explanations, show=False)
        plt.savefig(output_dir / f"bar_fold{fold}.png")
        plt.close()


if __name__ == "__main__":
    interpret_tabpfn()
