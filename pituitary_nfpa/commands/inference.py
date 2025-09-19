"""Perform DL model inference."""
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import cast

import click
from pycrumbs import tracked
import monai
import mlflow
import polars as pl
import pandas as pd
import lightning as pyl
from sklearn.metrics import roc_auc_score
import torch

from pituitary_nfpa import locations
from pituitary_nfpa.constants import VAL_RATE, MLFLOW_CLASSIFIER_PROJECT_NAME
from pituitary_nfpa.data import prepare_dataset
from pituitary_nfpa.preprocess.loaders import get_transforms
from pituitary_nfpa.train.modules import PituitaryClassifier


@click.command()
@click.argument("model-name")
@tracked(
    literal_directory=locations.checkpoints_dir,
    subdirectory_name_parameter="model_name",
    directory_injection_parameter="model_dir",
)
def inference(
    model_name: str,
    model_dir: Path,
) -> dict[str, float]:
    """Run inference with a trained model.

    This will write results parquet file into the existing checkpoints directory.

    MODEL_NAME is a free text string that names the model, also the directory
    within the project checkpoints directory where the model checkpoint is
    stored.

    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    # Set download location to avoid home directory for large model weights
    os.environ["HF_HUB_CACHE"] = str(locations.huggingface_cache_dir)

    config_file = model_dir / "train_config.json"

    # Read in the config file
    with config_file.open("r") as jf:
        config = json.load(jf)

    train_record_file = model_dir / "train_record.json"
    with train_record_file.open("r") as jf:
        train_record = json.load(jf)

    fold_str: str | int | None = config["fold"]
    # Need to get the fold from the train record
    if fold_str is None:
        fold_str = train_record["environment"]["environment_variables"][
            "SLURM_ARRAY_TASK_ID"
        ]
    fold = int(fold_str)

    train_data, test_data = prepare_dataset(
        target=config["target"],
        fold=fold,
    )

    val_cutoff = int((1.0 - VAL_RATE) * len(train_data))
    val_data = train_data[val_cutoff:]
    train_data = train_data[:val_cutoff]

    use_gpu_preprocessing = config["use_gpu_preprocessing"]
    test_transform = get_transforms(
        augmentation=False,
        use_cache=False,
        use_gpu=use_gpu_preprocessing,
    )

    train_dataset = monai.data.Dataset(
        train_data,
        test_transform,
    )
    val_dataset = monai.data.Dataset(
        val_data,
        test_transform,
    )
    test_dataset = monai.data.Dataset(
        test_data,
        test_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        prefetch_factor=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=config["workers"],
        batch_size=config["batch_size"],
        prefetch_factor=4,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        num_workers=config["workers"],
        batch_size=config["batch_size"],
        prefetch_factor=4,
    )

    # Get the checkpoint file. Due to a misunderstanding, there are two
    # checkpoint directories for each model. One that mlflow logging created,
    # and another by pytorch lightning. We want the mlflow one, since that
    # stored the "best" val_auroc checkpoint, whereas the copy in the main main
    # directory is the latest (and these models overfit badly))
    # To get the location of the mlflow version, we need to do some mlflow
    # queries
    original_model_name = train_record["called_function"]["parameters"]["model_name"]
    original_model_name += f"-{fold:02}"
    mlflow.set_tracking_uri(locations.mlflow_tracking_uri)
    mlflow.set_registry_uri(locations.mlflow_registry_uri)
    mlflow.set_experiment(MLFLOW_CLASSIFIER_PROJECT_NAME)
    runs_df = cast(pd.DataFrame, mlflow.search_runs())
    runs_df = runs_df[runs_df["tags.mlflow.runName"] == original_model_name]  # type: ignore

    if len(runs_df) != 1:
        raise RuntimeError(f"Found {len(runs_df)} runs matching this name")

    artifact_dir = Path(runs_df.iloc[0].artifact_uri)
    ckpt_path = artifact_dir / "checkpoints/latest_checkpoint.pth"
    module = PituitaryClassifier.load_from_checkpoint(ckpt_path, config=config)

    trainer = pyl.Trainer(
        accelerator="gpu",
        default_root_dir=model_dir,
        log_every_n_steps=1,
        max_epochs=config["epochs"],
    )

    split_results = {}

    for split_name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        results: dict[str, list] = {
            "mrn": [],
            "label": [],
            "accession": [],
            "prediction": [],
        }

        predictions = trainer.predict(module, loader)

        for batch in predictions:  # type: ignore
            batch = cast(dict, batch)
            results["mrn"].extend(batch["mrn"])
            results["accession"].extend(batch["accession"])
            results["label"].extend(batch["label"].tolist())
            results["prediction"].extend(batch["prediction"].flatten().tolist())

        auroc = roc_auc_score(y_true=results["label"], y_score=results["prediction"])
        print(split_name, auroc)

        df = pl.DataFrame(results)
        out_file = model_dir / f"{split_name}_inferences.parquet"
        df.write_parquet(out_file)

        split_results[split_name] = auroc

    return split_results


if __name__ == "__main__":
    inference()
