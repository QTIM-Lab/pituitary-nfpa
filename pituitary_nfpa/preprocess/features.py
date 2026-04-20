import numpy as np
import polars as pl


N_CLINICAL_FEATURES = 12


def get_clinical_features(
    df: pl.DataFrame,
    impute_missing: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Get formatted clinical features.

    Given a copy of the working dataframe, extract clinical features in a
    form suitable to pass to a model.

    Returns
    -------
    features: np.ndarray
        Array of features
    feature_names: list[str]
        List of feature names.

    """
    columns = []
    names = []

    # Sex (binary)
    columns.append(df["sex"].to_numpy())
    names.append("sex")

    # Age - divide by 100
    columns.append(df["age"].to_numpy() / 100.0)
    names.append("age")

    # Cavernous Sinus Invasion (binary)
    columns.append(df["cs_invasion"].to_numpy())
    names.append("age")

    # Ki-67: needs imputation
    if impute_missing:
        df = df.with_columns(pl.col("ki67").fill_null(strategy="mean"))
    columns.append(df["ki67"].to_numpy())
    names.append("ki67")

    # Size - scale by 50
    columns.append(df["size"].to_numpy() / 50.0)
    names.append("size")

    # IHC columns - between 0 by 4
    for f in ["acth", "lh", "fsh", "alpha_subunit", "gh", "prl", "tsh"]:
        if impute_missing:
            df = df.with_columns(pl.col(f).fill_null(strategy="mean"))
        columns.append(df[f].to_numpy() / 4.0)
        names.append(f)

    return np.column_stack(columns), names
