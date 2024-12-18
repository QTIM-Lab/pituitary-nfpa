"""Main training functions."""
import inspect
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Optional

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
from monai.metrics.rocauc import ROCAUCMetric

from pituitary_nfpa import locations
from pituitary_nfpa.constants import (
    IMAGE_KEY,
    LABEL_KEY,
    MLFLOW_CLASSIFER_PROJECT_NAME,
)
from pituitary_nfpa.data import prepare_dataset
from pituitary_nfpa.preprocess.loaders import get_transforms


class PituitaryClassifier(pyl.LightningModule):
    """Pytorch lightning module for the pituitary classifier."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initializer.

        Parameters
        ----------
        config: dict[str, Any]:
            Dictionary containing configuration parameters.

        """
        super().__init__()
        self.config = config

        self.model = monai.networks.nets.resnet10(
            spatial_dims=3,
            num_classes=1,
            n_input_channels=1,
        )
        self.train_auroc = ROCAUCMetric()
        self.val_auroc = ROCAUCMetric()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Pytorch lightning configure optimizers step."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.config["patience"],
            min_lr=self.config["lr"] / 100.0,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Pytorch lightning training step."""
        pred = self.model(batch[IMAGE_KEY])
        label = batch[LABEL_KEY][:, None]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            label,
        )
        self.log("train_loss", loss)  # type: ignore
        self.train_auroc.extend(pred.detach(), label)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Pytorch lightning validation step."""
        pred = self.model(batch[IMAGE_KEY])
        label = batch[LABEL_KEY][:, None]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            label,
        )
        self.log("val_loss", loss)  # type: ignore
        self.val_auroc.extend(pred, label)
        return loss

    def on_train_epoch_end(self) -> None:
        """Pytorch lightning epoch end step."""
        auroc = self.train_auroc.aggregate()
        self.log("train_auroc", auroc)  # type: ignore
        self.train_auroc.reset()
        super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        """Pytorch lightning epoch end step."""
        auroc = self.val_auroc.aggregate()
        self.log("val_auroc", auroc)  # type: ignore
        self.val_auroc.reset()
        super().on_validation_epoch_end()


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

    train_data, test_data = prepare_dataset(
        target=config["target"],
        fold=config["fold"],
    )

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

    train_dataset = monai.data.CacheDataset(train_data, train_transform)
    val_dataset = monai.data.CacheDataset(test_data, val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=True,
        prefetch_factor=4,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=config["workers"],
        batch_size=config["batch_size"],
        prefetch_factor=4,
    )

    mlflow.set_registry_uri(locations.mlflow_registry_uri)
    mlflow.set_tracking_uri(locations.mlflow_tracking_uri)

    # Create the project if it doesn't exist already
    try:
        mlflow.create_experiment(
            name=MLFLOW_CLASSIFER_PROJECT_NAME,
            artifact_location=str(locations.mlflow_artifacts_dir),
        )
    except mlflow.MlflowException:
        pass

    mlflow.set_experiment(experiment_name=MLFLOW_CLASSIFER_PROJECT_NAME)
    mlflow.pytorch.autolog()

    lr_monitor = pyl.pytorch.callbacks.LearningRateMonitor()

    module = PituitaryClassifier(config=config)
    trainer = pyl.Trainer(
        accelerator="gpu",
        default_root_dir=model_output_dir,
        log_every_n_steps=1,
        max_epochs=config["epochs"],
        callbacks=[lr_monitor],
    )

    with mlflow.start_run():

        py_module = inspect.getmodule(PituitaryClassifier)

        if py_module is None:
            raise RuntimeError("Cannot get module information.")

        mlflow.log_params(config)
        mlflow.log_params(get_environment_info())
        mlflow.log_params(get_git_info(py_module, allow_dirty=True))
        mlflow.log_params(get_installed_packages())
        mlflow.log_params(get_environment_vars())

        trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    train()
