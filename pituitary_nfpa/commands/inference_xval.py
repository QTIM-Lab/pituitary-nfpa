import click
import polars as pl

from pituitary_nfpa import locations
from pituitary_nfpa.commands.inference import inference


@click.command()
@click.argument("model-name")
def inference_xval(model_name: str) -> None:
    """Perform inference over an experiment with multiple folds."""
    fold_results = []
    for model_dir in locations.checkpoints_dir.glob(f"{model_name}_*"):
        fold_name = model_dir.stem
        fold_results.append(inference.callback(fold_name))  # type: ignore

    df = pl.DataFrame(fold_results)
    print(df.mean())
    print(df.std())


if __name__ == "__main__":
    inference_xval()
