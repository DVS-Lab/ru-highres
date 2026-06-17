from __future__ import annotations

import pandas as pd

from openneuro_voxels.aggregate import (
    create_dataset_resolution_summary,
    create_sensitivity_summary,
    create_single_band_rankings,
    create_single_band_summary,
)


def test_duplicate_subjects_runs_echoes_collapse_to_one_resolution() -> None:
    frame = pd.DataFrame(
        [
            _row("ds1", "sub-01_task-rest_run-1_echo-1_bold.nii.gz", "01", "rest", 1, 2.4),
            _row("ds1", "sub-01_task-rest_run-1_echo-2_bold.nii.gz", "01", "rest", 2, 2.4),
            _row("ds1", "sub-02_task-rest_run-2_echo-1_bold.nii.gz", "02", "rest", 1, 2.4),
        ]
    )

    summary = create_dataset_resolution_summary(frame)

    assert len(summary) == 1
    assert summary.loc[0, "n_files"] == 3
    assert summary.loc[0, "n_unique_subjects"] == 2
    assert summary.loc[0, "resting_state_present"]


def test_multiple_tasks_at_different_resolutions_create_two_rows() -> None:
    frame = pd.DataFrame(
        [
            _row("ds1", "sub-01_task-rest_bold.nii.gz", "01", "rest", None, 2.4),
            _row("ds1", "sub-01_task-nback_bold.nii.gz", "01", "nback", None, 3.0),
        ]
    )

    summary = create_dataset_resolution_summary(frame)

    assert len(summary) == 2
    assert set(summary["task_labels"]) == {"rest", "nback"}


def test_sensitivity_excludes_singletons_and_small_groups() -> None:
    summary = pd.DataFrame(
        [
            {"n_files": 1, "percent_dataset_bold_files": 50.0},
            {"n_files": 2, "percent_dataset_bold_files": 4.9},
            {"n_files": 2, "percent_dataset_bold_files": 5.0},
        ]
    )

    sensitivity = create_sensitivity_summary(summary)

    assert len(sensitivity) == 1


def test_single_band_summary_excludes_multiband_and_unknown() -> None:
    frame = pd.DataFrame(
        [
            _row("ds1", "sub-01_task-a_bold.nii.gz", "01", "a", None, 2.0, "single"),
            _row("ds2", "sub-01_task-b_bold.nii.gz", "01", "b", None, 2.0, "multiband"),
            _row("ds3", "sub-01_task-c_bold.nii.gz", "01", "c", None, 2.0, "unknown"),
        ]
    )

    summary = create_dataset_resolution_summary(frame)
    single_band = create_single_band_summary(summary)
    rankings = create_single_band_rankings(summary, reference_lr_mm=1.75, reference_ap_mm=1.75)

    assert single_band["dataset_accession"].tolist() == ["ds1"]
    assert set(rankings["axis"]) == {"lr", "ap"}
    assert rankings["n_dataset_resolution_rows"].tolist() == [1, 1]


def _row(
    dataset: str,
    key: str,
    subject: str,
    task: str,
    echo: int | None,
    resolution: float,
    acquisition_type: str = "single",
) -> dict[str, object]:
    acquisition_type_value = {
        "single": "likely_single_band_or_conventional_epi",
        "multiband": "multiband_or_sms",
        "unknown": "unknown",
    }[acquisition_type]
    return {
        "dataset_accession": dataset,
        "key": f"{dataset}/sub-{subject}/func/{key}",
        "subject": subject,
        "session": None,
        "task": task,
        "acquisition": None,
        "reconstruction": None,
        "direction": None,
        "run": None,
        "echo": str(echo) if echo is not None else None,
        "part": None,
        "metadata_status": "found" if acquisition_type != "unknown" else "missing",
        "metadata_source_keys": (),
        "metadata_json": {},
        "acquisition_type": acquisition_type_value,
        "acquisition_type_confidence": "test",
        "multiband_acceleration_factor": 4.0 if acquisition_type == "multiband" else None,
        "slice_acceleration_factor": None,
        "inplane_acceleration_factor": None,
        "repetition_time_s": 2.0,
        "echo_time_s": 0.028,
        "magnetic_field_strength_t": 3.0,
        "manufacturer": "Siemens",
        "pulse_sequence_type": None,
        "scanning_sequence": None,
        "sequence_name": None,
        "protocol_name": None,
        "axis_mapping_successful": True,
        "canonical_voxel_size_lr_mm": resolution,
        "canonical_voxel_size_ap_mm": resolution,
        "canonical_voxel_size_is_mm": resolution,
        "native_voxel_size_x_mm": resolution,
        "native_voxel_size_y_mm": resolution,
        "native_voxel_size_z_mm": resolution,
        "voxel_size_lr_mm": resolution,
        "voxel_size_ap_mm": resolution,
        "voxel_size_is_mm": resolution,
        "qc_flags": (),
    }
