from __future__ import annotations

from openneuro_voxels.bids_entities import is_raw_bold_key, parse_bids_entities


def test_raw_bold_filter_accepts_func_bold() -> None:
    key = "ds000001/sub-01/ses-02/func/sub-01_ses-02_task-rest_run-1_bold.nii.gz"

    assert is_raw_bold_key(key)


def test_raw_bold_filter_rejects_excluded_and_nonbold_paths() -> None:
    rejected = [
        "ds000001/derivatives/sub-01/func/sub-01_task-rest_bold.nii.gz",
        "ds000001/sub-01/anat/sub-01_T1w.nii.gz",
        "ds000001/sub-01/func/sub-01_task-rest_sbref.nii.gz",
        "ds000001/sub-01/func/sub-01_task-rest_bold.dtseries.nii",
        "ds000001/.git/annex/objects/file",
    ]

    assert not any(is_raw_bold_key(key) for key in rejected)


def test_parse_entities_and_rest_label() -> None:
    entities = parse_bids_entities(
        "ds123/sub-05/ses-a/func/"
        "sub-05_ses-a_task-rest_acq-highres_dir-AP_run-2_echo-1_part-mag_bold.nii.gz"
    )

    assert entities.subject == "05"
    assert entities.session == "a"
    assert entities.task == "rest"
    assert entities.acquisition == "highres"
    assert entities.direction == "AP"
    assert entities.run == "2"
    assert entities.echo == "1"
    assert entities.part == "mag"
    assert entities.is_resting_state
