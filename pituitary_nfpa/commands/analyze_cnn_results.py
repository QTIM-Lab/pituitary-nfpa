import click
import polars as pl

import numpy as np
from pituitary_nfpa import locations
import matplotlib.pyplot as plt
from sklearn import metrics


BOOTSTRAP_NUM = 1000


MODEL_NAMES = [
    ("basic-cnn-scratch-2025-11-07", "Image-Only CNN"),
    ("multimodal-cnn-scratch-2025-11-07", "Multi-modal CNN"),
]


@click.command()
def analyze_cnn_results() -> None:
    """Analyze results over an experiment with multiple folds."""
    output_dir = locations.vis_dir / "cnn_rocs"
    output_dir.mkdir(exist_ok=True)

    for model_name, display_name in MODEL_NAMES:
        fold_results = []
        for model_dir in locations.checkpoints_dir.glob(f"{model_name}_*"):
            fold_results.append(pl.read_parquet(model_dir / "test_inferences.parquet"))

        df = pl.concat(fold_results)

        plt.figure(figsize=(6, 6))

        auroc = metrics.roc_auc_score(
            y_score=df["prediction"],
            y_true=df["label"],
        )
        fpr, tpr, _ = metrics.roc_curve(
            y_score=df["prediction"],
            y_true=df["label"],
        )
        plt.plot(fpr, tpr, label=f"{display_name} ({auroc:.3f})")

        bootstrap_aurocs = []

        for _ in range(BOOTSTRAP_NUM):
            bootstrap_df = df.sample(len(df), with_replacement=True)

            bootstrap_aurocs.append(
                metrics.roc_auc_score(
                    y_score=bootstrap_df["prediction"],
                    y_true=bootstrap_df["label"],
                )
            )

        # Find 95 confidence interval
        lower, upper = np.percentile(bootstrap_aurocs, (2.5, 97.5))

        print(f"{display_name}: {auroc:.3f} ({lower:.3f}-{upper:.3f})")

        plt.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            label="Chance level",
            color="black",
            linestyle="dashed",
        )
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(display_name)
        plt.legend()
        plt.savefig(output_dir / "roc.png")


if __name__ == "__main__":
    analyze_cnn_results()
