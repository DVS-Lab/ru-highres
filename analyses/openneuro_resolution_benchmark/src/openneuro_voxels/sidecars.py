"""BIDS sidecar metadata helpers for acquisition-type stratification."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, cast

from botocore.client import BaseClient
from botocore.exceptions import ClientError

from openneuro_voxels.bids_entities import BidsEntities

_MULTIBAND_TEXT_RE = re.compile(r"(multi[-_ ]?band|sms|(^|[_-])mb[0-9]+|[_-]mb[_-])", re.I)


@dataclass(frozen=True)
class AcquisitionMetadata:
    metadata_status: str
    metadata_source_keys: tuple[str, ...]
    metadata_json: dict[str, Any]
    acquisition_type: str
    acquisition_type_confidence: str
    multiband_acceleration_factor: float | None
    slice_acceleration_factor: float | None
    inplane_acceleration_factor: float | None
    repetition_time_s: float | None
    echo_time_s: float | None
    magnetic_field_strength_t: float | None
    manufacturer: str | None
    pulse_sequence_type: str | None
    scanning_sequence: str | None
    sequence_name: str | None
    protocol_name: str | None

    def as_database_values(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata_source_keys"] = json.dumps(payload["metadata_source_keys"])
        payload["metadata_json"] = json.dumps(payload["metadata_json"], sort_keys=True)
        return payload


def load_acquisition_metadata(
    client: BaseClient,
    *,
    bucket: str,
    key: str,
    entities: BidsEntities,
) -> AcquisitionMetadata:
    """Fetch and merge available BIDS sidecar JSON metadata for one BOLD object."""

    merged: dict[str, Any] = {}
    sources: list[str] = []
    for candidate in sidecar_candidates(key, entities):
        try:
            response = client.get_object(Bucket=bucket, Key=candidate)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                continue
            raise
        payload = cast(bytes, response["Body"].read())
        try:
            metadata = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(metadata, dict):
            merged.update(metadata)
            sources.append(candidate)

    if not sources:
        return classify_acquisition_metadata({}, metadata_status="missing", source_keys=())
    return classify_acquisition_metadata(
        merged, metadata_status="found", source_keys=tuple(dict.fromkeys(sources))
    )


def classify_acquisition_metadata(
    metadata: dict[str, Any],
    *,
    metadata_status: str = "found",
    source_keys: tuple[str, ...] = (),
) -> AcquisitionMetadata:
    """Classify likely BOLD acquisition type from BIDS sidecar fields."""

    multiband = _as_float(metadata.get("MultibandAccelerationFactor"))
    slice_accel = _as_float(metadata.get("SliceAccelerationFactor"))
    inplane = _as_float(
        metadata.get("ParallelReductionFactorInPlane")
        or metadata.get("ParallelReductionFactorInPlaneRetired")
    )

    text_values = [
        _as_str(metadata.get("PulseSequenceType")),
        _as_str(metadata.get("ScanningSequence")),
        _as_str(metadata.get("SequenceName")),
        _as_str(metadata.get("ProtocolName")),
        _as_str(metadata.get("ImageType")),
    ]
    text = " ".join(value for value in text_values if value)
    explicit_mb = any(value is not None and value > 1 for value in (multiband, slice_accel))
    text_mb = bool(_MULTIBAND_TEXT_RE.search(text))

    if metadata_status != "found":
        acquisition_type = "unknown"
        confidence = "no_bids_sidecar_metadata"
    elif explicit_mb or text_mb:
        acquisition_type = "multiband_or_sms"
        confidence = "explicit_factor" if explicit_mb else "sequence_text"
    else:
        acquisition_type = "likely_single_band_or_conventional_epi"
        confidence = "metadata_without_multiband_evidence"

    return AcquisitionMetadata(
        metadata_status=metadata_status,
        metadata_source_keys=source_keys,
        metadata_json=metadata,
        acquisition_type=acquisition_type,
        acquisition_type_confidence=confidence,
        multiband_acceleration_factor=multiband,
        slice_acceleration_factor=slice_accel,
        inplane_acceleration_factor=inplane,
        repetition_time_s=_as_float(metadata.get("RepetitionTime")),
        echo_time_s=_as_float(metadata.get("EchoTime")),
        magnetic_field_strength_t=_as_float(metadata.get("MagneticFieldStrength")),
        manufacturer=_as_str(metadata.get("Manufacturer")),
        pulse_sequence_type=_as_str(metadata.get("PulseSequenceType")),
        scanning_sequence=_as_str(metadata.get("ScanningSequence")),
        sequence_name=_as_str(metadata.get("SequenceName")),
        protocol_name=_as_str(metadata.get("ProtocolName")),
    )


def sidecar_candidates(key: str, entities: BidsEntities) -> tuple[str, ...]:
    """Return plausible BIDS sidecar JSON keys from general to specific."""

    path = PurePosixPath(key)
    parts = path.parts
    if "func" not in parts:
        return ()

    func_index = parts.index("func")
    dataset = parts[0]
    directories = [PurePosixPath(dataset)]
    if entities.subject:
        directories.append(PurePosixPath(dataset) / f"sub-{entities.subject}")
    if entities.subject and entities.session:
        directories.append(
            PurePosixPath(dataset) / f"sub-{entities.subject}" / f"ses-{entities.session}"
        )
    directories.append(PurePosixPath(*parts[: func_index + 1]))

    filenames = _candidate_filenames(path.name, entities)
    candidates: list[str] = []
    for directory in directories:
        for filename in filenames:
            candidates.append(str(directory / filename))
    return tuple(dict.fromkeys(candidates))


def _candidate_filenames(filename: str, entities: BidsEntities) -> tuple[str, ...]:
    exact_stem = _strip_nifti_extension(filename)
    entity_parts = []
    if entities.task:
        entity_parts.append(f"task-{entities.task}")
    if entities.acquisition:
        entity_parts.append(f"acq-{entities.acquisition}")
    if entities.reconstruction:
        entity_parts.append(f"rec-{entities.reconstruction}")
    if entities.direction:
        entity_parts.append(f"dir-{entities.direction}")

    names = ["bold.json"]
    if entities.task:
        names.append(f"task-{entities.task}_bold.json")
    if entity_parts:
        names.append("_".join([*entity_parts, "bold.json"]))
    names.append(f"{exact_stem}.json")
    return tuple(dict.fromkeys(names))


def _strip_nifti_extension(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return "\\".join(str(item) for item in value)
    return str(value)
