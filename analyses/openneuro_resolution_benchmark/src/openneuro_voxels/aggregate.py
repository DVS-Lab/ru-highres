"""Dataset-resolution aggregation from the SQLite audit store."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from openneuro_voxels.config import OutputConfig


def load_file_level_headers(database: Path) -> pd.DataFrame:
    """Return one row per successfully parsed BOLD file."""

    connection = sqlite3.connect(database)
    try:
        query = """
            SELECT
                o.dataset_accession,
                o.key,
                o.size,
                o.etag,
                o.last_modified,
                o.subject,
                o.session,
                o.task,
                o.acquisition,
                o.reconstruction,
                o.direction,
                o.run,
                o.echo,
                o.part,
                o.scan_status,
                o.compressed_bytes_retrieved,
                o.retry_count,
                o.extraction_timestamp,
                o.software_version,
                o.git_commit,
                o.metadata_status,
                o.metadata_source_keys,
                o.metadata_json,
                o.acquisition_type,
                o.acquisition_type_confidence,
                o.multiband_acceleration_factor,
                o.slice_acceleration_factor,
                o.inplane_acceleration_factor,
                o.repetition_time_s,
                o.echo_time_s,
                o.magnetic_field_strength_t,
                o.manufacturer,
                o.pulse_sequence_type,
                o.scanning_sequence,
                o.sequence_name,
                o.protocol_name,
                h.header_type,
                h.image_shape,
                h.ndim,
                h.n_volumes,
                h.native_voxel_size_x_mm,
                h.native_voxel_size_y_mm,
                h.native_voxel_size_z_mm,
                h.qform_code,
                h.sform_code,
                h.best_affine,
                h.orientation_codes,
                h.voxel_size_lr_mm,
                h.voxel_size_ap_mm,
                h.voxel_size_is_mm,
                h.axis_mapping_successful,
                h.canonical_voxel_size_lr_mm,
                h.canonical_voxel_size_ap_mm,
                h.canonical_voxel_size_is_mm,
                h.qc_flags
            FROM s3_objects o
            JOIN nifti_headers h ON h.key = o.key
        """
        frame = pd.read_sql_query(query, connection)
    finally:
        connection.close()

    if frame.empty:
        return frame
    frame["image_shape"] = frame["image_shape"].map(_loads_json)
    frame["best_affine"] = frame["best_affine"].map(_loads_json)
    frame["orientation_codes"] = frame["orientation_codes"].map(_loads_json)
    frame["qc_flags"] = frame["qc_flags"].map(lambda value: tuple(_loads_json(value) or []))
    frame["metadata_source_keys"] = frame["metadata_source_keys"].map(
        lambda value: tuple(_loads_json(value) or [])
    )
    frame["metadata_json"] = frame["metadata_json"].map(_loads_json)
    frame["resting_state"] = frame["task"].eq("rest")
    return frame


def create_dataset_resolution_summary(
    file_headers: pd.DataFrame,
    *,
    config: OutputConfig | None = None,
) -> pd.DataFrame:
    """Create one row per dataset accession x unique anatomical-axis triplet."""

    config = config or OutputConfig()
    if file_headers.empty:
        return _empty_summary()

    primary = file_headers[
        file_headers["axis_mapping_successful"].astype(bool)
        & file_headers[
            [
                "canonical_voxel_size_lr_mm",
                "canonical_voxel_size_ap_mm",
                "canonical_voxel_size_is_mm",
            ]
        ]
        .notna()
        .all(axis=1)
    ].copy()
    if primary.empty:
        return _empty_summary()

    dataset_totals = (
        file_headers.groupby("dataset_accession")["key"].count().rename("dataset_bold_files")
    )
    group_cols = [
        "dataset_accession",
        "canonical_voxel_size_lr_mm",
        "canonical_voxel_size_ap_mm",
        "canonical_voxel_size_is_mm",
    ]
    rows: list[dict[str, Any]] = []
    for group_key, group in primary.groupby(group_cols, dropna=False):
        dataset = str(group_key[0])
        total = int(dataset_totals.loc[dataset])
        n_files = int(group["key"].count())
        qc_flags = sorted({flag for flags in group["qc_flags"] for flag in flags})
        tasks = sorted(value for value in group["task"].dropna().unique())
        rows.append(
            {
                "dataset_accession": dataset,
                "voxel_size_lr_mm": group_key[1],
                "voxel_size_ap_mm": group_key[2],
                "voxel_size_is_mm": group_key[3],
                "native_x_min_mm": group["native_voxel_size_x_mm"].min(),
                "native_x_max_mm": group["native_voxel_size_x_mm"].max(),
                "native_y_min_mm": group["native_voxel_size_y_mm"].min(),
                "native_y_max_mm": group["native_voxel_size_y_mm"].max(),
                "native_z_min_mm": group["native_voxel_size_z_mm"].min(),
                "native_z_max_mm": group["native_voxel_size_z_mm"].max(),
                "lr_min_unrounded_mm": group["voxel_size_lr_mm"].min(),
                "lr_max_unrounded_mm": group["voxel_size_lr_mm"].max(),
                "ap_min_unrounded_mm": group["voxel_size_ap_mm"].min(),
                "ap_max_unrounded_mm": group["voxel_size_ap_mm"].max(),
                "is_min_unrounded_mm": group["voxel_size_is_mm"].min(),
                "is_max_unrounded_mm": group["voxel_size_is_mm"].max(),
                "n_files": n_files,
                "n_unique_subjects": group["subject"].nunique(dropna=True),
                "n_unique_sessions": group["session"].nunique(dropna=True),
                "task_labels": ";".join(tasks),
                "resting_state_present": bool(group["task"].eq("rest").any()),
                "non_resting_tasks_present": bool(group["task"].dropna().ne("rest").any()),
                "acquisition_labels": _join_unique(group["acquisition"]),
                "run_labels": _join_unique(group["run"]),
                "echo_labels": _join_unique(group["echo"]),
                "acquisition_type_labels": _join_unique(group["acquisition_type"]),
                "likely_single_band_present": bool(
                    group["acquisition_type"].eq("likely_single_band_or_conventional_epi").any()
                ),
                "multiband_or_sms_present": bool(
                    group["acquisition_type"].eq("multiband_or_sms").any()
                ),
                "unknown_acquisition_type_present": bool(
                    group["acquisition_type"].isna().any()
                    or group["acquisition_type"].eq("unknown").any()
                ),
                "metadata_status_labels": _join_unique(group["metadata_status"]),
                "multiband_acceleration_factors": _join_unique_numeric(
                    group["multiband_acceleration_factor"]
                ),
                "slice_acceleration_factors": _join_unique_numeric(
                    group["slice_acceleration_factor"]
                ),
                "inplane_acceleration_factors": _join_unique_numeric(
                    group["inplane_acceleration_factor"]
                ),
                "repetition_times_s": _join_unique_numeric(group["repetition_time_s"]),
                "dataset_bold_files": total,
                "percent_dataset_bold_files": (100.0 * n_files / total) if total else 0.0,
                "representative_s3_key": group["key"].sort_values().iloc[0],
                "singleton_group": n_files == 1,
                "qc_flags": ";".join(qc_flags),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["dataset_accession", "voxel_size_lr_mm", "voxel_size_ap_mm", "voxel_size_is_mm"]
    )


def create_native_axis_summary(file_headers: pd.DataFrame) -> pd.DataFrame:
    """Summarize files whose affine could not be mapped to anatomical axes."""

    if file_headers.empty:
        return pd.DataFrame()
    native = file_headers[~file_headers["axis_mapping_successful"].astype(bool)].copy()
    if native.empty:
        return pd.DataFrame()
    cols = [
        "dataset_accession",
        "native_voxel_size_x_mm",
        "native_voxel_size_y_mm",
        "native_voxel_size_z_mm",
    ]
    rows = []
    dataset_totals = (
        file_headers.groupby("dataset_accession")["key"].count().rename("dataset_bold_files")
    )
    for group_key, group in native.groupby(cols, dropna=False):
        dataset = str(group_key[0])
        total = int(dataset_totals.loc[dataset])
        rows.append(
            {
                "dataset_accession": dataset,
                "native_voxel_size_x_mm": round(float(group_key[1]), 3),
                "native_voxel_size_y_mm": round(float(group_key[2]), 3),
                "native_voxel_size_z_mm": round(float(group_key[3]), 3),
                "n_files": int(group["key"].count()),
                "percent_dataset_bold_files": 100.0 * int(group["key"].count()) / total,
                "representative_s3_key": group["key"].sort_values().iloc[0],
                "qc_flags": ";".join(
                    sorted({flag for flags in group["qc_flags"] for flag in flags})
                ),
            }
        )
    return pd.DataFrame(rows)


def create_sensitivity_summary(
    summary: pd.DataFrame,
    *,
    min_files: int = 2,
    min_dataset_percent: float = 5.0,
) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    return summary[
        (summary["n_files"] >= min_files)
        & (summary["percent_dataset_bold_files"] >= min_dataset_percent)
    ].copy()


def create_common_triplets(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "voxel_size_lr_mm",
                "voxel_size_ap_mm",
                "voxel_size_is_mm",
                "n_dataset_resolution_rows",
                "n_datasets",
            ]
        )
    cols = ["voxel_size_lr_mm", "voxel_size_ap_mm", "voxel_size_is_mm"]
    grouped = summary.groupby(cols, dropna=False)
    return (
        grouped.agg(
            n_dataset_resolution_rows=("dataset_accession", "count"),
            n_datasets=("dataset_accession", "nunique"),
        )
        .reset_index()
        .sort_values(["n_dataset_resolution_rows", "n_datasets"], ascending=False)
    )


def create_single_band_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset-resolution rows to likely single-band/conventional EPI data."""

    if summary.empty or "acquisition_type_labels" not in summary:
        return summary.iloc[0:0].copy()
    return summary[
        summary["likely_single_band_present"].astype(bool)
        & ~summary["multiband_or_sms_present"].astype(bool)
        & ~summary["unknown_acquisition_type_present"].astype(bool)
    ].copy()


def create_single_band_rankings(
    summary: pd.DataFrame,
    *,
    reference_lr_mm: float = 1.75,
    reference_ap_mm: float = 1.75,
) -> pd.DataFrame:
    """Rank the study's in-plane voxel size against likely single-band OpenNeuro rows."""

    single_band = create_single_band_summary(summary)
    rows: list[dict[str, Any]] = []
    for axis, column, reference in [
        ("lr", "voxel_size_lr_mm", reference_lr_mm),
        ("ap", "voxel_size_ap_mm", reference_ap_mm),
    ]:
        values = single_band[column].dropna()
        n = int(values.shape[0])
        if n == 0:
            percent_finer_or_equal = None
            percent_coarser_or_equal = None
            n_finer_or_equal = 0
            n_finer = 0
            n_equal = 0
            n_coarser = 0
            n_coarser_or_equal = 0
            rank = None
        else:
            n_finer_or_equal = int((values <= reference).sum())
            n_finer = int((values < reference).sum())
            n_equal = int((values == reference).sum())
            n_coarser = int((values > reference).sum())
            n_coarser_or_equal = int((values >= reference).sum())
            rank = n_finer + 1
            percent_finer_or_equal = 100.0 * n_finer_or_equal / n
            percent_coarser_or_equal = 100.0 * n_coarser_or_equal / n
        rows.append(
            {
                "reference_label": "ru_highres_in_plane",
                "axis": axis,
                "reference_voxel_size_mm": reference,
                "comparison_group": "likely_single_band_or_conventional_epi",
                "n_dataset_resolution_rows": n,
                "reference_rank_when_smaller_is_better": rank,
                "n_rows_finer_than_reference": n_finer,
                "n_rows_equal_to_reference": n_equal,
                "n_rows_coarser_than_reference": n_coarser,
                "n_rows_finer_or_equal_to_reference": n_finer_or_equal,
                "n_rows_coarser_or_equal_to_reference": n_coarser_or_equal,
                "percent_rows_finer_or_equal_to_reference": percent_finer_or_equal,
                "percent_rows_coarser_or_equal_to_reference": percent_coarser_or_equal,
            }
        )
    return pd.DataFrame(rows)


def write_aggregation_outputs(
    *,
    database: Path,
    config: OutputConfig | None = None,
) -> dict[str, Path]:
    config = config or OutputConfig()
    config.results_dir.mkdir(parents=True, exist_ok=True)
    file_headers = load_file_level_headers(database)
    summary = create_dataset_resolution_summary(file_headers, config=config)
    sensitivity = create_sensitivity_summary(
        summary,
        min_files=config.sensitivity_min_files,
        min_dataset_percent=config.sensitivity_min_dataset_percent,
    )
    native = create_native_axis_summary(file_headers)
    common = create_common_triplets(summary)
    single_band = create_single_band_summary(summary)
    single_band_rankings = create_single_band_rankings(summary)

    outputs = {
        "file_level_headers_parquet": config.results_dir / "file_level_headers.parquet",
        "dataset_resolution_summary_csv": config.results_dir
        / "dataset_resolution_summary.csv",
        "dataset_resolution_summary_parquet": config.results_dir
        / "dataset_resolution_summary.parquet",
        "dataset_resolution_sensitivity_csv": config.results_dir
        / "dataset_resolution_sensitivity.csv",
        "dataset_resolution_native_summary_csv": config.results_dir
        / "dataset_resolution_native_summary.csv",
        "dataset_resolution_single_band_summary_csv": config.results_dir
        / "dataset_resolution_single_band_summary.csv",
        "single_band_reference_rankings_csv": config.results_dir
        / "single_band_reference_rankings.csv",
        "common_resolution_triplets_csv": config.results_dir / "common_resolution_triplets.csv",
    }

    file_headers.to_parquet(outputs["file_level_headers_parquet"], index=False)
    summary.to_csv(outputs["dataset_resolution_summary_csv"], index=False)
    summary.to_parquet(outputs["dataset_resolution_summary_parquet"], index=False)
    sensitivity.to_csv(outputs["dataset_resolution_sensitivity_csv"], index=False)
    native.to_csv(outputs["dataset_resolution_native_summary_csv"], index=False)
    single_band.to_csv(outputs["dataset_resolution_single_band_summary_csv"], index=False)
    single_band_rankings.to_csv(outputs["single_band_reference_rankings_csv"], index=False)
    common.to_csv(outputs["common_resolution_triplets_csv"], index=False)
    return outputs


def _join_unique(series: pd.Series) -> str:
    return ";".join(sorted(str(value) for value in series.dropna().unique()))


def _join_unique_numeric(series: pd.Series) -> str:
    values = sorted({float(value) for value in series.dropna().unique()})
    return ";".join(f"{value:g}" for value in values)


def _loads_json(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dataset_accession",
            "voxel_size_lr_mm",
            "voxel_size_ap_mm",
            "voxel_size_is_mm",
            "n_files",
            "n_unique_subjects",
            "n_unique_sessions",
            "task_labels",
            "resting_state_present",
            "non_resting_tasks_present",
            "acquisition_labels",
            "run_labels",
            "echo_labels",
            "dataset_bold_files",
            "percent_dataset_bold_files",
            "representative_s3_key",
            "singleton_group",
            "qc_flags",
        ]
    )
