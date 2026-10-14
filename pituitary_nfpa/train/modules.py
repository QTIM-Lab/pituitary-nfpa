from typing import Any

import monai
import lightning as pyl
import torch
from monai.metrics.rocauc import ROCAUCMetric

from pituitary_nfpa.constants import (
    CLINICAL_DATA_KEY,
    IMAGE_KEY,
    LABEL_KEY,
)
from pituitary_nfpa.preprocess.features import N_CLINICAL_FEATURES


CLINICAL_MODEL_WIDTH = 32


class JointFusionModel(torch.nn.Module):
    """Joint fusion model with two feature encoders and a head."""

    def __init__(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        head: torch.nn.Module,
    ) -> None:
        """Initializer.

        Parameters
        ----------
        model1: torch.nn.Module
            Encoder for the first input.
        model2: torch.nn.Module
            Encoder for the second input.
        head: torch.nn.Module
            Head, takes as input the outputs of the model1 and model2,
            concatenated down axis 1 and produces the output.

        """
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.head = head

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Forward method.

        Parameters
        ----------
        input1: torch.Tensor
            Input to the first model.
        input2: torch.Tensor
            Input to the second model.

        Returns
        -------
        output: torch.Tensor
            Output of the head

        """
        f_1 = self.model1(input1)
        f_2 = self.model2(input2)
        f_all = torch.concat([f_1, f_2], dim=1)
        return self.head(f_all)


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
            spatial_dims=2 if config["single_slice"] else 3,
            num_classes=1,
            n_input_channels=1,
            feed_forward=False,
            bias_downsample=False,
            pretrained=config["pretrained"],
        )

        n_features = 512  # features at output of model
        use_clinical = config.get("use_clinical", False)
        if use_clinical:
            n_features += CLINICAL_MODEL_WIDTH

        head = torch.nn.Linear(n_features, 1)
        head.bias.data = torch.tensor(0.0)

        self.model: torch.nn.Module

        if use_clinical:
            clinical_model = torch.nn.Sequential(
                torch.nn.Linear(N_CLINICAL_FEATURES, CLINICAL_MODEL_WIDTH),
                torch.nn.SiLU(),
                torch.nn.Linear(CLINICAL_MODEL_WIDTH, CLINICAL_MODEL_WIDTH),
                torch.nn.SiLU(),
                torch.nn.Linear(CLINICAL_MODEL_WIDTH, CLINICAL_MODEL_WIDTH),
                torch.nn.SiLU(),
            )
            self.model = JointFusionModel(backbone, clinical_model, head)
        else:
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
        pred = self.forward_tensors(batch)
        label = batch[LABEL_KEY][:, None]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            label,
        )
        self.log("train_loss", loss, batch_size=len(pred))  # type: ignore
        self.train_auroc.extend(pred.detach(), label)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Pytorch lightning validation step."""
        pred = self.forward_tensors(batch)
        label = batch[LABEL_KEY][:, None]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            label,
        )
        self.log("val_loss", loss, batch_size=len(pred))  # type: ignore
        self.val_auroc.extend(pred, label)
        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        """Pytorch lightning test step."""
        pred = self.forward_tensors(batch)
        label = batch[LABEL_KEY][:, None]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred,
            label,
        )
        self.log("test_loss", loss, batch_size=len(pred))  # type: ignore
        self.test_auroc.extend(pred, label)

    def forward(self, x: dict[str, Any]) -> dict[str, Any]:
        """Forward step, run a batch and return results."""
        x["prediction"] = self.forward_tensors(x)
        return x

    def forward_tensors(self, x: dict[str, Any]) -> torch.Tensor:
        """Produces the prediction tensor from input dictionary."""
        if self.config.get("use_clinical", False):
            return self.model(x[IMAGE_KEY], x[CLINICAL_DATA_KEY])

        return self.model(x[IMAGE_KEY])

    def on_train_epoch_end(self) -> None:
        """Pytorch lightning epoch end step."""
        auroc = self.train_auroc.aggregate()
        self.log("train_auroc", auroc, on_epoch=True)  # type: ignore
        self.train_auroc.reset()
        super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        """Pytorch lightning epoch end step."""
        auroc = self.val_auroc.aggregate()
        self.log("val_auroc", auroc, on_epoch=True)  # type: ignore
        self.val_auroc.reset()
        super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        """Pytorch lightning epoch end step."""
        auroc = self.test_auroc.aggregate()
        self.log("test_auroc", auroc, on_epoch=True)  # type: ignore
        self.test_auroc.reset()
        super().on_test_epoch_end()
