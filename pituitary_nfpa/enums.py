"""Location to store enumerations associated with your project.

Enums should be used wherever there are a limited number of discrete
values for a certain variable. This is preferable to using simple
strings, since errors are likely to be found earlier, it is clear
what options are available, and enums may be iterated through.

"""
from enum import Enum


class DatasetSplit(Enum):
    """Enum listing partitions of the dataset."""

    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


class PredictionLabels(Enum):
    """Enum listing prediction targets."""

    PROGRESSION = "PROGRESSION"
    RECURRENCE = "RECURRENCE"
    PROGRESSION_OR_RECURRENCE = "PROGRESSION_OR_RECURRENCE"
