"""Perform DL model inference."""
import json
import os
from pathlib import Path

import click
from pycrumbs import tracked
import monai
import lightning as pyl

from pituitary_nfpa import locations
from pituitary_nfpa.constants import VAL_RATE
from pituitary_nfpa.data import prepare_dataset
from pituitary_nfpa.preprocess.loaders import get_transforms


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
    """Train a model.

    This will create a new directory in the checkpoints directory containing
    model weights and any other files associated with model training.

    CONFIG_FILE is the name of the config file within the repo's training
    config directory containing all parameters for the training job.

    MODEL_NAME is a free text string that names the model. The output will be
    placed in a directory with this name.

    """
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

    # train_dataset = monai.data.Dataset(
    #     train_data,
    #     test_transform,
    # )
    # val_dataset = monai.data.Dataset(
    #     val_data,
    #     test_transform,
    # )
    test_dataset = monai.data.Dataset(
        test_data,
        test_transform,
    )

    trainer = pyl.Trainer(
        accelerator="gpu",
        default_root_dir=model_dir,
        log_every_n_steps=1,
        max_epochs=config["epochs"],
    )

    # train_predictions = trainer.predict(train_dataset, ckpt_path="best")
    # val_predictions = trainer.predict(val_dataset, ckpt_path="best")
    test_predictions = trainer.predict(test_dataset, ckpt_path="best")
    print(type(test_predictions))
    print(test_predictions)


if __name__ == "__main__":
    inference()
