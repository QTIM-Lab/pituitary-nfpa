"""Fit and evaluate radiomic models."""
import json
from pathlib import Path
import sys
from typing import Optional

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
from sklearn.feature_selection import SelectKBest, f_classif

import mlflow
import tabpfn

from pituitary_nfpa import locations
from pituitary_nfpa.preprocess.features import get_clinical_features
from pituitary_nfpa.constants import (
    MLFLOW_RADIOMIC_PROJECT_NAME,
    VAL_RATE,
    N_FOLDS,
)


@click.command()
@click.argument("model_name")
@click.option("--seed", "-s", type=int, help="Random seed to use.")
@click.option("--features", "-f", type=int, help="Number of features to select.")
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
@click.option(
    "--cnn-prediction-model",
    "-m",
    help="Model name to load CNN predictions to include in the model",
)
@tracked(
    literal_directory=locations.checkpoints_dir,
    subdirectory_name_parameter="model_name",
    seed_parameter="seed",
    include_uuid=True,
    directory_injection_parameter="model_output_dir",
    seed_torch=False,
)
def fit_tabular(
    model_name: str,
    model_output_dir: Path,
    use_clinical: bool,
    use_radiomics: bool,
    cnn_prediction_model: str | None = None,
    seed: Optional[int] = None,
    features: Optional[int] = None,
) -> None:
    """Train a tabular model.

    Inputs are some combination of clinical, radiomic, and CNN predictions.

    This will create a new directory in the checkpoints directory containing
    model weights and any other files associated with model training.

    MODEL_NAME is a free text string that names the model. The output will be
    placed in a directory with this name.

    """
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

    fold_mapping = {}
    if cnn_prediction_model is not None:
        model_dirs = list(locations.checkpoints_dir.glob(f"{cnn_prediction_model}_*"))
        if len(model_dirs) != N_FOLDS:
            raise ValueError(f"Found {len(model_dirs)} matching the pattern")

        for d in model_dirs:
            record_file = d / "train_record.json"

            with record_file.open("r") as jf:
                train_record = json.load(jf)

            fold = int(
                train_record["environment"]["environment_variables"][
                    "SLURM_ARRAY_TASK_ID"
                ]
            )

            fold_mapping[fold] = d

        if len(fold_mapping) != N_FOLDS:
            raise ValueError("Did not find one model per fold")

    per_fold_val_aurocs: dict[str, list[float]] = {}
    per_fold_test_aurocs: dict[str, list[float]] = {}

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
            feature_cols = [c for c in features_df.columns if c.startswith("original")]
            train_features_list.append(train_df[feature_cols].to_numpy())
            val_features_list.append(val_df[feature_cols].to_numpy())
            test_features_list.append(test_df[feature_cols].to_numpy())
        if use_clinical:
            train_features_list.append(get_clinical_features(train_df))
            val_features_list.append(get_clinical_features(val_df))
            test_features_list.append(get_clinical_features(test_df))
        if cnn_prediction_model is not None:
            d = fold_mapping[fold]
            train_inferences_csv = d / "train_inferences.parquet"
            train_inferences = pl.read_parquet(train_inferences_csv)
            joined_inferences = train_df.join(
                train_inferences,
                how="left",
                on=("mrn", "accession"),
                maintain_order="left",
                validate="1:1",
            )
            train_features_list.append(joined_inferences["prediction"].to_numpy())
            val_inferences_csv = d / "val_inferences.parquet"
            val_inferences = pl.read_parquet(val_inferences_csv)
            joined_inferences = val_df.join(
                val_inferences,
                how="left",
                on=("mrn", "accession"),
                maintain_order="left",
                validate="1:1",
            )
            val_features_list.append(joined_inferences["prediction"].to_numpy())
            test_inferences_csv = d / "test_inferences.parquet"
            test_inferences = pl.read_parquet(test_inferences_csv)
            joined_inferences = test_df.join(
                test_inferences,
                how="left",
                on=("mrn", "accession"),
                maintain_order="left",
                validate="1:1",
            )
            test_features_list.append(joined_inferences["prediction"].to_numpy())

        train_features = np.column_stack(train_features_list)
        val_features = np.column_stack(val_features_list)
        test_features = np.column_stack(test_features_list)

        target_col = "progression_or_recurrence"
        train_labels = np.squeeze(train_df[[target_col]].to_numpy(), 1)
        test_labels = np.squeeze(test_df[[target_col]].to_numpy(), 1)
        val_labels = np.squeeze(val_df[[target_col]].to_numpy(), 1)

        if features is not None:
            selector = SelectKBest(f_classif, k=features)
            selector.fit(train_features, train_labels)
            train_features = selector.transform(train_features)
            val_features = selector.transform(val_features)
            test_features = selector.transform(test_features)

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

        for estimator in [
            sklearn.linear_model.LogisticRegression(max_iter=10000),
            sklearn.ensemble.RandomForestClassifier(),
            sklearn.ensemble.AdaBoostClassifier(),
            sklearn.ensemble.GradientBoostingClassifier(),
            tabpfn.TabPFNClassifier(),
        ]:
            estimator_name = estimator.__class__.__name__

            if estimator_name not in per_fold_val_aurocs:
                per_fold_val_aurocs[estimator_name] = []
                per_fold_test_aurocs[estimator_name] = []

            run_name = f"{model_name}-{fold:02d}-{estimator_name}"

            with mlflow.start_run(run_name=run_name):

                py_module = sys.modules[__name__]

                if py_module is None:
                    raise RuntimeError("Cannot get module information.")

                mlflow.log_params(
                    {
                        "fold_used": fold,
                        "estimator": estimator_name,
                        "n_features": features,
                        "use_clinical": use_clinical,
                        "use_radiomics": use_radiomics,
                        "cnn_prediction_model": use_radiomics,
                    }
                )
                mlflow.log_params(get_environment_info())
                mlflow.log_params(get_git_info(py_module, allow_dirty=True))
                mlflow.log_params(get_installed_packages())
                mlflow.log_params(get_environment_vars())

                kwargs = {}
                if not isinstance(estimator, tabpfn.TabPFNClassifier):
                    kwargs["sample_weight"] = train_sample_weights

                estimator.fit(
                    train_features,
                    train_labels,
                    **kwargs,
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
                mlflow.log_metric("val_auroc", val_auroc)
                mlflow.log_metric("val_f1", val_f1)
                per_fold_val_aurocs[estimator_name].append(val_auroc)

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
                mlflow.log_metric("test_auroc", test_auroc)
                mlflow.log_metric("test_f1", test_f1)
                per_fold_test_aurocs[estimator_name].append(test_auroc)

    print("Validation AUROCs")
    for estimator_name, aurocs in per_fold_val_aurocs.items():
        avg = np.mean(aurocs).item()
        print(f"{estimator_name}: {avg:.3f}")

    print("Test AUROCs")
    for estimator_name, aurocs in per_fold_test_aurocs.items():
        avg = np.mean(aurocs).item()
        print(f"{estimator_name}: {avg:.3f}")


if __name__ == "__main__":
    fit_tabular()
