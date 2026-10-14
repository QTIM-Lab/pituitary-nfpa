import json
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
@click.argument("display_name")
@tracked(
    directory_parameter="model_directory",
)
def analyze_results(
    model_directory: Path,
    display_name: str,
) -> None:
    """Analyze pre-computed results.

    MODEL_DIRECTORY is the location of a directory containing the model output.

    """
    predictions_file = model_directory / "test_predictions.parquet"
    predictions_df = pl.read_parquet(predictions_file).with_columns(
        pl.col("label").cast(pl.Boolean),
        (pl.col("score") > 0.5).alias("binary_pred"),
    )

    plt.figure(figsize=(6, 6))

    metrics_dict = {}

    # Calculate pooled AUROCs, with bootstrapping
    for (estimator,), est_df in predictions_df.group_by(
        "estimator",
        maintain_order=True,
    ):
        auroc = metrics.roc_auc_score(
            y_score=est_df["score"],
            y_true=est_df["label"],
        )
        fpr, tpr, th = metrics.roc_curve(
            y_score=est_df["score"],
            y_true=est_df["label"],
        )
        youden_threshold = th[np.argmax(tpr - fpr)]

        est_df = est_df.with_columns(
            (pl.col("score") > youden_threshold).alias("binary_pred"),
        )
        auprc = metrics.average_precision_score(
            y_score=est_df["score"],
            y_true=est_df["label"],
        )
        classification_report = metrics.classification_report(
            y_pred=est_df["binary_pred"],
            y_true=est_df["label"],
            output_dict=True,
        )
        f1 = classification_report["True"]["f1-score"]
        precision = classification_report["True"]["precision"]
        recall = classification_report["True"]["recall"]
        npv = classification_report["False"]["precision"]
        specificity = classification_report["False"]["recall"]
        accuracy = classification_report["accuracy"]
        plt.plot(fpr, tpr, label=f"{estimator} ({auroc:.3f})")

        bootstrap_auroc = []
        bootstrap_f1 = []
        bootstrap_precision = []
        bootstrap_recall = []
        bootstrap_npv = []
        bootstrap_specificity = []
        bootstrap_accuracy = []
        bootstrap_auprc = []

        for _ in range(BOOTSTRAP_NUM):
            bootstrap_df = est_df.sample(len(est_df), with_replacement=True)

            bootstrap_auroc.append(
                metrics.roc_auc_score(
                    y_score=bootstrap_df["score"],
                    y_true=bootstrap_df["label"],
                )
            )
            classification_report = metrics.classification_report(
                y_pred=bootstrap_df["binary_pred"],
                y_true=bootstrap_df["label"],
                output_dict=True,
            )
            bootstrap_f1.append(classification_report["True"]["f1-score"])
            bootstrap_precision.append(classification_report["True"]["precision"])
            bootstrap_recall.append(classification_report["True"]["recall"])
            bootstrap_npv.append(classification_report["False"]["precision"])
            bootstrap_specificity.append(classification_report["False"]["recall"])
            bootstrap_accuracy.append(classification_report["accuracy"])
            bootstrap_auprc.append(
                metrics.average_precision_score(
                    y_score=bootstrap_df["score"],
                    y_true=bootstrap_df["label"],
                )
            )

        # Find 95 confidence interval
        auroc_lower, auroc_upper = np.percentile(bootstrap_auroc, (2.5, 97.5))
        print(f"{estimator}: {auroc:.3f} ({auroc_lower:.3f}-{auroc_upper:.3f})")

        auprc_lower, auprc_upper = np.percentile(bootstrap_auprc, (2.5, 97.5))
        recall_lower, recall_upper = np.percentile(bootstrap_recall, (2.5, 97.5))
        precision_lower, precision_upper = np.percentile(
            bootstrap_precision, (2.5, 97.5)
        )
        specificity_lower, specificity_upper = np.percentile(
            bootstrap_specificity, (2.5, 97.5)
        )
        npv_lower, npv_upper = np.percentile(bootstrap_npv, (2.5, 97.5))
        f1_lower, f1_upper = np.percentile(bootstrap_f1, (2.5, 97.5))
        accuracy_lower, accuracy_upper = np.percentile(bootstrap_accuracy, (2.5, 97.5))

        metrics_dict[estimator] = {
            "auroc": (auroc, auroc_lower, auroc_upper),
            "auprc": (auprc, auprc_lower, auprc_upper),
            "recall": (recall, recall_lower, recall_upper),
            "precision": (precision, precision_lower, precision_upper),
            "specificity": (specificity, specificity_lower, specificity_upper),
            "npv": (npv, npv_lower, npv_upper),
            "f1": (f1, f1_lower, f1_upper),
            "accuracy": (accuracy, accuracy_lower, accuracy_upper),
        }
        print(metrics_dict[estimator])

    plt.plot(
        [0.0, 1.0], [0.0, 1.0], label="Chance level", color="black", linestyle="dashed"
    )
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(display_name)
    plt.legend()
    plt.savefig(model_directory / "roc.png")

    with model_directory.joinpath("metrics.json").open("w") as f:
        json.dump(metrics_dict, f, indent=4)


if __name__ == "__main__":
    analyze_results()
