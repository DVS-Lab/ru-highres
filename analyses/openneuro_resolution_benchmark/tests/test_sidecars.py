from __future__ import annotations

from openneuro_voxels.bids_entities import parse_bids_entities
from openneuro_voxels.sidecars import classify_acquisition_metadata, sidecar_candidates


def test_sidecar_candidates_include_inheritance_and_exact_json() -> None:
    key = (
        "ds000001/sub-01/ses-mri/func/"
        "sub-01_ses-mri_task-rest_acq-highres_run-1_bold.nii.gz"
    )
    entities = parse_bids_entities(key)

    candidates = sidecar_candidates(key, entities)

    assert "ds000001/task-rest_bold.json" in candidates
    assert "ds000001/sub-01/ses-mri/func/task-rest_acq-highres_bold.json" in candidates
    assert (
        "ds000001/sub-01/ses-mri/func/"
        "sub-01_ses-mri_task-rest_acq-highres_run-1_bold.json"
    ) in candidates


def test_multiband_factor_classifies_as_multiband() -> None:
    metadata = classify_acquisition_metadata({"MultibandAccelerationFactor": 4})

    assert metadata.acquisition_type == "multiband_or_sms"
    assert metadata.acquisition_type_confidence == "explicit_factor"


def test_sequence_text_classifies_as_multiband() -> None:
    metadata = classify_acquisition_metadata({"ProtocolName": "task_mb4_sms"})

    assert metadata.acquisition_type == "multiband_or_sms"
    assert metadata.acquisition_type_confidence == "sequence_text"


def test_metadata_without_multiband_evidence_is_likely_single_band() -> None:
    metadata = classify_acquisition_metadata(
        {
            "RepetitionTime": 2.0,
            "ParallelReductionFactorInPlane": 2,
            "SequenceName": "epfid2d1_128",
        }
    )

    assert metadata.acquisition_type == "likely_single_band_or_conventional_epi"
    assert metadata.inplane_acceleration_factor == 2


def test_missing_metadata_is_unknown() -> None:
    metadata = classify_acquisition_metadata({}, metadata_status="missing")

    assert metadata.acquisition_type == "unknown"
    assert metadata.acquisition_type_confidence == "no_bids_sidecar_metadata"
