# This is a standalone script that should be executed in the
# pituitary-nfpa-radiomics environment (separate from the main project)
from pathlib import Path
import polars as pl
from radiomics import featureextractor

from pycrumbs.track import tracked


@tracked(directory_parameter="output_dir")
def extract_features(output_dir):
    df_path = "/autofs/space/crater_001/projects/pituitary-nfpa/meta/working_dataframe.parquet"
    df = pl.read_parquet(df_path)
    df = df.filter(pl.col("include"))

    extractor = featureextractor.RadiomicsFeatureExtractor()

    results = []

    for row in df.iter_rows(named=True):
        try:
            features = extractor.execute(row["image_path"], row["seg_path"])
        except ValueError:
            print(row["seg_path"])

        # Remove the unneeded "diagnostics" features
        used_features = {
            k: v.item()
            for k, v in features.items()
            if k.startswith("original")
        }

        # Copy basic id information from the original csv
        for k in ["case_number", "mrn", "accession"]:
            used_features[k] = row[k]

        results.append(used_features)

    results_df = pl.DataFrame(results)
    features_path = Path(output_dir) / "radiomic_features.parquet"
    results_df.write_parquet(features_path)


if __name__ == "__main__":
    extract_features("/autofs/space/crater_001/projects/pituitary-nfpa/meta")
