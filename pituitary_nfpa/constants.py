"""Location to store various constants used in the project.

Python does not have true constants, but these values should be treated
as constant by other modules. By convention, constants use ALL_CAPS.

"""
IMAGE_KEY = "image"
MASK_KEY = "mask"
LABEL_KEY = "label"

N_FOLDS = 6
MIN_FOLLOWUP_MONTHS = 3

MLFLOW_CLASSIFIER_PROJECT_NAME = "pituitary-classifier-2025-08"
MLFLOW_RADIOMIC_PROJECT_NAME = "pituitary-radiomic-classifier-2025-08"

VAL_RATE = 0.2
