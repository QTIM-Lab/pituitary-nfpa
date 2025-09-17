import click

from pituitary_nfpa import locations
from pituitary_nfpa.commands.inference import inference


@click.command()
@click.argument("model-name")
def inference_xval(model_name: str) -> None:
    """Perform inference over an experiment with multiple folds."""
    for model_dir in locations.checkpoints_dir.glob(f"{model_name}_*"):
        fold_name = model_dir.stem
        inference.callback(fold_name)  # type: ignore


if __name__ == "__main__":
    inference_xval()
