"""Main training functions."""
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Optional

import click
from pycrumbs import tracked
from pycrumbs.track import (
    get_environment_info,
    get_environment_vars,
    get_installed_packages,
    get_git_info,
)

import mlflow
import monai
import lightning as pyl
import torch

from pituitary_nfpa import locations
from pituitary_nfpa.constants import (
    LABEL_KEY,
    MLFLOW_CLASSIFIER_PROJECT_NAME,
    VAL_RATE,
)
from pituitary_nfpa.data import prepare_dataset
from pituitary_nfpa.preprocess.loaders import get_transforms
from pituitary_nfpa.train.modules import PituitaryClassifier


@click.command()
@click.argument("config_file")
@click.argument("model_name")
@click.option("--seed", "-s", type=int, help="Random seed to use.")
@tracked(
    literal_directory=locations.checkpoints_dir,
    subdirectory_name_parameter="model_name",
    seed_parameter="seed",
    include_uuid=True,
    directory_injection_parameter="model_output_dir",
)
def train(
    config_file: str,
    model_name: str,
    model_output_dir: Path,
    seed: Optional[int] = None,
) -> None:
    """Train a model.

    This will create a new directory in the checkpoints directory containing
    model weights and any other files associated with model training.

    CONFIG_FILE is the name of the config file within the repo's training
    config directory containing all parameters for the training job.

    MODEL_NAME is a free text string that names the model. The output will be
    placed in a directory with this name.

    """
    mp.set_start_method("spawn")

    # Set download location to avoid home directory for large model weights
    os.environ["HF_HUB_CACHE"] = str(locations.huggingface_cache_dir)

    # Read in the config file
    if not config_file.lower().endswith(".json"):
        config_file += ".json"
    config_path = locations.train_configs_dir / config_file
    with config_path.open("r") as jf:
        config = json.load(jf)

    # Stash the configuration in the output directory
    config_copy = model_output_dir / "train_config.json"
    with config_copy.open("w") as jf:
        json.dump(config, jf, indent=4)

    fold_str: str | int | None = config["fold"]
    if fold_str is None:
        fold_str = os.environ.get("SLURM_ARRAY_TASK_ID")
        if fold_str is None:
            raise RuntimeError("Cannot determine fold.")
    fold = int(fold_str)

    train_data, test_data = prepare_dataset(
        target=config["target"],
        fold=fold,
    )

    val_cutoff = int((1.0 - VAL_RATE) * len(train_data))
    val_data = train_data[val_cutoff:]
    train_data = train_data[:val_cutoff]

    use_cache = config["use_cache"]
    use_gpu_preprocessing = config["use_gpu_preprocessing"]
    train_transform = get_transforms(
        augmentation=True,
        use_cache=use_cache,
        use_gpu=use_gpu_preprocessing,
    )
    val_transform = get_transforms(
        augmentation=False,
        use_cache=use_cache,
        use_gpu=use_gpu_preprocessing,
    )

    train_dataset = monai.data.CacheDataset(
        train_data,
        train_transform,
        cache_rate=1.0 if use_cache else 0.0,
    )
    val_dataset = monai.data.CacheDataset(
        val_data,
        val_transform,
        cache_rate=1.0 if use_cache else 0.0,
    )
    test_dataset = monai.data.Dataset(
        test_data,
        val_transform,
    )

    # Calculate weights for a training sampler to approximately correct class
    # balance
    num_positive = sum(bool(item[LABEL_KEY]) for item in train_data)
    num_negative = len(train_data) - num_positive
    pos_weight = 0.5 / num_positive
    neg_weight = 0.5 / num_negative
    train_sample_weights = [
        pos_weight if item[LABEL_KEY] else neg_weight for item in train_data
    ]
    train_sampler = torch.utils.data.WeightedRandomSampler(
        train_sample_weights, len(train_data)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        sampler=train_sampler,
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

    mlflow.set_registry_uri(locations.mlflow_registry_uri)
    mlflow.set_tracking_uri(locations.mlflow_tracking_uri)

    # Create the project if it doesn't exist already
    try:
        mlflow.create_experiment(
            name=MLFLOW_CLASSIFIER_PROJECT_NAME,
            artifact_location=str(locations.mlflow_artifacts_dir),
        )
    except mlflow.MlflowException:
        pass

    mlflow.set_experiment(experiment_name=MLFLOW_CLASSIFIER_PROJECT_NAME)
    mlflow.enable_system_metrics_logging()
    mlflow.pytorch.autolog(
        log_models=True,
        checkpoint=True,
        checkpoint_monitor="val_auroc",
        checkpoint_mode="max",
    )

    lr_monitor = pyl.pytorch.callbacks.LearningRateMonitor()

    module = PituitaryClassifier(config=config)
    trainer = pyl.Trainer(
        accelerator="gpu",
        default_root_dir=model_output_dir,
        log_every_n_steps=1,
        max_epochs=config["epochs"],
        callbacks=[lr_monitor],
    )

    run_name = f"{model_name}-{fold:02d}"

    with mlflow.start_run(run_name=run_name):

        py_module = inspect.getmodule(PituitaryClassifier)

        if py_module is None:
            raise RuntimeError("Cannot get module information.")

        mlflow.log_params(config)
        mlflow.log_params({"fold_used": fold})
        mlflow.log_params(get_environment_info())
        mlflow.log_params(get_git_info(py_module, allow_dirty=True))
        mlflow.log_params(get_installed_packages())
        mlflow.log_params(get_environment_vars())

        trainer.fit(module, train_loader, val_loader)

        trainer.test(ckpt_path="best", dataloaders=test_loader)


if __name__ == "__main__":
    train()
