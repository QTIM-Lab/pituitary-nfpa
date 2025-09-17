"""Perform DL model inference."""
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import cast

import click
from pycrumbs import tracked
import monai
import polars as pl
import lightning as pyl
import torch

from pituitary_nfpa import locations
from pituitary_nfpa.constants import VAL_RATE
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
) -> None:
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

    fold_str: str | int | None = config["fold"]
    if fold_str is None:
        # Need to load in the train record to get the fold
        train_record_file = model_dir / "train_record.json"
        with train_record_file.open("r") as jf:
            train_record = json.load(jf)
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

    ckpt_path = next(
        (model_dir / "lightning_logs" / "version_0" / "checkpoints").glob("*.ckpt")
    )
    module = PituitaryClassifier.load_from_checkpoint(ckpt_path, config=config)

    trainer = pyl.Trainer(
        accelerator="gpu",
        default_root_dir=model_dir,
        log_every_n_steps=1,
        max_epochs=config["epochs"],
    )

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

        df = pl.DataFrame(results)
        out_file = model_dir / f"{split_name}_inferences.parquet"
        df.write_parquet(out_file)


if __name__ == "__main__":
    inference()
