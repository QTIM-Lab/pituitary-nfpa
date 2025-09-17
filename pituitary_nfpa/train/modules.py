from typing import Any

import monai
import lightning as pyl
import torch
from monai.metrics.rocauc import ROCAUCMetric

from pituitary_nfpa.constants import (
    IMAGE_KEY,
    LABEL_KEY,
)


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

        backbone = monai.networks.nets.resnet10(
            spatial_dims=3,
            num_classes=1,
            n_input_channels=1,
            feed_forward=False,
            bias_downsample=False,
            pretrained=config["pretrained"],
        )
        head = torch.nn.Linear(512, 1)
        head.bias.data = torch.tensor(0.0)
        self.model = torch.nn.Sequential(backbone, head)
        self.train_auroc = ROCAUCMetric()
        self.val_auroc = ROCAUCMetric()
        self.test_auroc = ROCAUCMetric()

    def configure_optimizers(self) -> dict[str, Any]:  # type: ignore
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

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Pytorch lightning test step."""
        pred = self.model(batch[IMAGE_KEY])
        label = batch[LABEL_KEY][:, None]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            label,
        )
        self.log("test_loss", loss)  # type: ignore
        self.test_auroc.extend(pred, label)

    def forward(self, x: dict[str, Any]) -> dict[str, Any]:
        """Forward step, run a batch and return results."""
        x["prediction"] = self.model(x[IMAGE_KEY])
        return x

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

    def on_test_epoch_end(self) -> None:
        """Pytorch lightning epoch end step."""
        auroc = self.test_auroc.aggregate()
        self.log("test_auroc", auroc)  # type: ignore
        self.test_auroc.reset()
        super().on_test_epoch_end()
