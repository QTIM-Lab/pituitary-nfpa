"""Check the imaging files to ensure all relevant data exists."""
import click
import polars as pl
from pycrumbs import tracked

import SimpleITK as sitk
from sklearn.model_selection import StratifiedKFold

from pituitary_nfpa import locations
from pituitary_nfpa.constants import N_FOLDS


FEATURE_COLS = {
    "age": "age",
    "sex 0=F,1=M": "sex",
    "tumor size (max diameter)": "size",
    "CS invasion": "cs_invasion",
    "Ki-67(%)": "ki67",
    "IHC staining 0vs1": "ihc",
    "ACTH": "acth",
}


@click.command()
@tracked(
    literal_directory=locations.working_dataframe.parent,
    seed_torch=False,
)
def check_dataset() -> None:
    """Find usable items in dataset and write out a simplified parquet with paths."""
    df = pl.read_excel(locations.dataset_workbook, sheet_name='Sheet1')
    df = df[
        [
            "case number",
            "MRN",
            "Accession # of pre-op MRI scan",
            "progression",
            "recurrence",
            "Segmentation Done",
            "Segmentation Slice",
            *FEATURE_COLS.keys()
        ]
    ]
    df = df.drop_nulls("case number")
    df = df.rename(FEATURE_COLS)

    missing_directory_count = 0
    missing_image_count = 0
    ambiguous_image_count = 0
    missing_segmentation_count = 0
    empty_segmentation_count = 0
    ok_count = 0

    output_list = []

    for case, mrn, acc, progression, recurrence, have_seg, desc, *features in df.iter_rows():
        if desc is not None:
            desc = desc.replace(":", "").replace("+", "")

        output_row = {
            "case_number": case,
            "mrn": mrn.strip(),
            "accession": acc.strip(),
            "progression": progression or 0,
            "recurrence": recurrence or 0,
            "progression_or_recurrence": progression or recurrence or 0,
            "exclude_reason": None,
            **{k: v for k, v in zip(FEATURE_COLS.values(), features)}
        }

        if have_seg is None or have_seg != "yes":
            output_row["exclude_reason"] = "Segmentation not performed"
            output_list.append(output_row)
            continue

        patient_dir = locations.images_dir / mrn
        if not patient_dir.exists():
            missing_directory_count += 1
            print("Missing directory", patient_dir)
            output_row["exclude_reason"] = "No image directory found"
            output_list.append(output_row)
            continue

        seg_files = [
            f for f in patient_dir.glob("*.seg.nrrd") if not f.name.startswith(".")
        ]
        if len(seg_files) < 1:
            missing_segmentation_count += 1
            print("Missing segmentation", patient_dir)
            output_row["exclude_reason"] = "No segmentation found"
            output_list.append(output_row)
            continue
        seg_path = seg_files[0]
        seg_im = sitk.ReadImage(str(seg_path))
        seg_arr = sitk.GetArrayFromImage(seg_im)
        if seg_arr.max() < 1:
            empty_segmentation_count += 1
            print("Empty segmentation", patient_dir)
            output_row["exclude_reason"] = "Segmentation is empty"
            output_list.append(output_row)
            continue

        im_files = [
            f
            for f in patient_dir.glob("*.nrrd")
            if not f.name.startswith(".") and ".seg.nrrd" not in f.name
            and "Segment" not in f.name
        ]
        if len(im_files) < 1:
            missing_image_count += 1
            print("Missing image", patient_dir, "found", len(im_files))
            output_row["exclude_reason"] = "No image found"
            output_list.append(output_row)
            continue
        elif len(im_files) > 1:
            # Found multiple image files. Choose the one that matches the
            # description in the excel file
            filtered_files = [f for f in im_files if desc in f.name]
            if len(filtered_files) < 1:
                ambiguous_image_count += 1
                print("Ambiguous image", patient_dir, "description", desc)
                for f in im_files:
                    print(f.name)
                output_row["exclude_reason"] = "Could not determine correct image"
                output_list.append(output_row)
                continue
            im_path = filtered_files[0]
        else:
            im_path = im_files[0]

        output_row["image_path"] = str(im_path)
        output_row["seg_path"] = str(seg_path)
        output_list.append(output_row)
        ok_count += 1

    print("Missing directory:", missing_directory_count)
    print("Missing image:", missing_image_count)
    print("Ambiguous image:", ambiguous_image_count)
    print("Missing segmentation:", missing_segmentation_count)
    print("Empty segmentation:", empty_segmentation_count)
    print("OK", ok_count)

    output_df = pl.DataFrame(output_list, infer_schema_length=None)
    output_df = output_df.with_columns(
        include=pl.col("exclude_reason").is_null(),
    )

    # Add splits
    positives = output_df.filter(pl.col("include"))
    negatives = output_df.filter(~pl.col("include"))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
    for fold_ind, (train_indices, test_indices) in enumerate(
        skf.split(
            positives["mrn"],
            positives["progression_or_recurrence"],
        )
    ):
        split = [
            "TRAIN" if i in train_indices else "TEST" for i in range(len(positives))
        ]
        split_col = pl.Series(f"fold_{fold_ind}", split)
        positives.insert_column(-1, split_col)

    output_df = pl.concat([positives, negatives], how="diagonal")

    output_df.write_parquet(locations.working_dataframe)
