"""Main training functions."""
import json
from pathlib import Path
from typing import Any, Optional

import click
from pycrumbs import tracked

import monai
import lightning as pyl
import torch

from pituitary_nfpa import locations
from pituitary_nfpa.constants import IMAGE_KEY, LABEL_KEY
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

        self.model = monai.networks.nets.resnet18(
            spatial_dims=3,
            num_classes=1,
            n_input_channels=1,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Pytorch lightning training step."""
        pred = self.model(batch[IMAGE_KEY])
        label = batch[LABEL_KEY][:, None]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            label,
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Pytorch lightning validation step."""
        pred = self.model(batch[IMAGE_KEY])
        label = batch[LABEL_KEY][:, None]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            label,
        )
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Pytorch lightning configure optimizers step."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        return optimizer


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

    train_transform = get_transforms(augmentation=True)
    val_transform = get_transforms(augmentation=False)

    train_dataset = monai.data.CacheDataset(train_data, train_transform)
    val_dataset = monai.data.CacheDataset(test_data, val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        num_workers=config["workers"],
        batch_size=config["batch_size"],
    )

    module = PituitaryClassifier(config=config)
    trainer = pyl.Trainer(
        accelerator="cpu",
        default_root_dir=model_output_dir,
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    train()
