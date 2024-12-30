"""Defines locations where project data are stored."""
import sys
from pathlib import Path
from warnings import warn

from importlib_resources import files

# Standard location for share mounting
# This will vary based on the platform
if sys.platform == "darwin":
    # 'darwin' just means MacOS
    # Linux servers and containers
    project_dir = Path("/autofs/space/crater_001/projects/pituitary-nfpa")
    project_dataset_dir = Path(
        "/autofs/space/crater_001/datasets/private/pituitary-nfpa"
    )
else:
    # Linux servers and containers
    project_dir = Path("/autofs/space/crater_001/projects/pituitary-nfpa")
    project_dataset_dir = Path(
        "/autofs/space/crater_001/datasets/private/pituitary-nfpa"
    )


# Issue a warning here if the project locations don't exist
if not project_dir.exists():
    warn(
        "The project directory is not found at the expected location: "
        f"{project_dir}.",
        UserWarning,
    )
if not project_dataset_dir.exists():
    warn(
        "The project dataset directory is not found at the expected location: "
        f"{project_dataset_dir}.",
        UserWarning,
    )

# Location to store metadata
meta_dir = project_dir / "meta"

# Location to store metadata
mlflow_dir = project_dir / "mlflow"
mlflow_artifacts_dir = project_dir / "artifacts"
mlflow_registry_dir = project_dir / "registry"
mlflow_db = mlflow_dir / "mlruns.db"
mlflow_tracking_uri = f"sqlite:///{str(mlflow_db.resolve())}"
mlflow_registry_uri = f"file:/{str(mlflow_registry_dir.resolve())}"

# Hugging face hub cache (locations where model weights are downloaded to)
huggingface_cache_dir = project_dir / "huggingface_cache"

# Location to store checkpoints
checkpoints_dir = project_dir / "checkpoints"

# Location to store inferences on train/test/val data
inferences_dir = project_dir / "inferences"

# Location to store results of analyses
analysis_dir = project_dir / "analysis"

# Location to store visualizations
vis_dir = project_dir / "vis"

# Location where any resources within the package codebase are installed
resources_dir = files("pituitary_nfpa") / "resources"
# Location where train configs are kept
train_configs_dir = resources_dir / "training_configs"

# Location of the master excel file
dataset_workbook = project_dataset_dir / "nfpa-dataset-no-header.xlsx"
working_dataframe = meta_dir / "working_dataframe.parquet"
images_dir = project_dataset_dir / "images"
