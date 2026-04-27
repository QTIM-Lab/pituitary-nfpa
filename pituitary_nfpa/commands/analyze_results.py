from pathlib import Path
from pycrumbs import tracked

import click
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn import metrics


BOOTSTRAP_NUM = 1000


@click.command()
@click.argument(
    "model_directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@tracked(
    directory_parameter="model_directory",
)
def analyze_results(
    model_directory: Path,
) -> None:
    """Analyze pre-computed results.

    MODEL_DIRECTORY is the location of a directory containing the model output.

    """
    predictions_file = model_directory / "test_predictions.parquet"
    predictions_df = pl.read_parquet(predictions_file)

    plt.figure(figsize=(5, 5))

    # Calculate pooled AUROCs, with bootstrapping
    for (estimator,), est_df in predictions_df.group_by(
        "estimator",
        maintain_order=True,
    ):
        auroc = metrics.roc_auc_score(
            y_score=est_df["score"],
            y_true=est_df["label"],
        )
        fpr, tpr, _ = metrics.roc_curve(
            y_score=est_df["score"],
            y_true=est_df["label"],
        )
        plt.plot(fpr, tpr, label=estimator)

        bootstrap_aurocs = []

        for _ in range(BOOTSTRAP_NUM):
            bootstrap_df = est_df.sample(len(est_df), with_replacement=True)

            bootstrap_aurocs.append(
                metrics.roc_auc_score(
                    y_score=bootstrap_df["score"],
                    y_true=bootstrap_df["label"],
                )
            )

        # Find 95 confidence interval
        lower, upper = np.percentile(bootstrap_aurocs, (2.5, 97.5))

        print(f"{estimator}: {auroc:.3f} ({lower:.3f}-{upper:.3f})")

    plt.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        label="Chance level",
        color="black",
        linestyle="dashed"
    )
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.savefig(model_directory / "roc.png")


if __name__ == "__main__":
    analyze_results()
