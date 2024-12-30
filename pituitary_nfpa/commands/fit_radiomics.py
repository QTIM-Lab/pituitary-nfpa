"""Fit and evaluate radiomic models."""
import os
from pathlib import Path
import sys
from typing import Any, Optional

import click
import numpy as np
import polars as pl
from pycrumbs import tracked
from pycrumbs.track import (
    get_environment_info,
    get_environment_vars,
    get_installed_packages,
    get_git_info,
)
import sklearn
from sklearn import metrics

import mlflow

from pituitary_nfpa import locations
from pituitary_nfpa.constants import (
    IMAGE_KEY,
    LABEL_KEY,
    MLFLOW_RADIOMIC_PROJECT_NAME,
    VAL_RATE,
    N_FOLDS,
)


@click.command()
@click.argument("model_name")
@click.option("--seed", "-s", type=int, help="Random seed to use.")
@tracked(
    literal_directory=locations.checkpoints_dir,
    subdirectory_name_parameter="model_name",
    seed_parameter="seed",
    include_uuid=True,
    directory_injection_parameter="model_output_dir",
    seed_torch=False,
)
def fit_radiomics(
    model_name: str,
    model_output_dir: Path,
    seed: Optional[int] = None,
) -> None:
    """Train a model.

    This will create a new directory in the checkpoints directory containing
    model weights and any other files associated with model training.

    MODEL_NAME is a free text string that names the model. The output will be
    placed in a directory with this name.

    """
    features_df = pl.read_parquet(locations.radiomic_features_dataframe)
    meta_df = pl.read_parquet(locations.working_dataframe)

    joined_df = features_df.join(
        meta_df,
        how='left',
        on=("mrn", "accession", "case_number"),
        validate="1:1",
    )

    for fold in range(N_FOLDS):

        train_df = joined_df.filter(pl.col(f"fold_{fold}") == "TRAIN")
        test_df = joined_df.filter(pl.col(f"fold_{fold}") == "TEST")

        val_cutoff = int((1.0 - VAL_RATE) * len(train_df))
        val_df = train_df[val_cutoff:]
        train_df = train_df[:val_cutoff]

        feature_cols = [
            c for c in features_df.columns if c.startswith("original")
        ]
        target_col = "progression_or_recurrence"

        train_features = train_df[feature_cols].to_numpy()
        train_labels = np.squeeze(train_df[[target_col]].to_numpy(), 1)
        val_features = val_df[feature_cols].to_numpy()
        val_labels = np.squeeze(val_df[[target_col]].to_numpy(), 1)
        test_features = test_df[feature_cols].to_numpy()
        test_labels = np.squeeze(test_df[[target_col]].to_numpy(), 1)

        # Calculate weights for training to approximately correct class balance
        num_positive = train_labels.sum()
        num_negative = len(train_labels) - num_positive
        pos_weight = 0.5 / num_positive
        neg_weight = 0.5 / num_negative
        train_sample_weights = [
            pos_weight if lab else neg_weight for lab in train_labels
        ]

        mlflow.set_registry_uri(locations.mlflow_registry_uri)
        mlflow.set_tracking_uri(locations.mlflow_tracking_uri)

        # Create the project if it doesn't exist already
        try:
            mlflow.create_experiment(
                name=MLFLOW_RADIOMIC_PROJECT_NAME,
                artifact_location=str(locations.mlflow_artifacts_dir),
            )
        except mlflow.MlflowException:
            pass

        mlflow.set_experiment(experiment_name=MLFLOW_RADIOMIC_PROJECT_NAME)
        mlflow.sklearn.autolog()

        run_name = f"{model_name}-{fold:02d}"

        with mlflow.start_run(run_name=run_name):

            py_module = sys.modules[__name__]

            if py_module is None:
                raise RuntimeError("Cannot get module information.")

            mlflow.log_params({"fold_used": fold})
            mlflow.log_params(get_environment_info())
            mlflow.log_params(get_git_info(py_module, allow_dirty=True))
            mlflow.log_params(get_installed_packages())
            mlflow.log_params(get_environment_vars())

            estimator = sklearn.ensemble.RandomForestClassifier(
                min_samples_split=0.05,
                min_samples_leaf=0.05,
            )

            estimator.fit(
                train_features,
                train_labels,
                sample_weight=train_sample_weights,
            )

            # Validation
            val_predictions = estimator.predict(val_features)
            val_probs = estimator.predict_proba(val_features)[:, 1]

            val_auroc = metrics.roc_auc_score(
                y_score=val_probs,
                y_true=val_labels,
            )
            val_f1 = metrics.f1_score(
                y_pred=val_predictions,
                y_true=val_labels,
            )
            mlflow.log_metric('val_auroc', val_auroc)
            mlflow.log_metric('val_f1', val_f1)

            # Testing
            test_predictions = estimator.predict(test_features)
            test_probs = estimator.predict_proba(test_features)[:, 1]

            test_auroc = metrics.roc_auc_score(
                y_score=test_probs,
                y_true=test_labels,
            )
            test_f1 = metrics.f1_score(
                y_pred=test_predictions,
                y_true=test_labels,
            )
            mlflow.log_metric('test_auroc', test_auroc)
            mlflow.log_metric('test_f1', test_f1)


if __name__ == "__main__":
    fit_radiomics()
