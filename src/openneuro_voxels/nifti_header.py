"""Partial NIfTI header parsing for S3 ranged responses."""

from __future__ import annotations

import io
import math
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import nibabel as nib
import numpy as np
from nibabel.orientations import io_orientation

NIFTI1_HEADER_SIZE = 348
NIFTI2_HEADER_SIZE = 540
MIN_UNCOMPRESSED_HEADER_BYTES = 560


class HeaderParseError(RuntimeError):
    """Raised when partial bytes cannot be interpreted as a NIfTI header."""


class InsufficientHeaderBytes(HeaderParseError):
    """Raised when a larger ranged request is needed."""


@dataclass(frozen=True)
class OrientationMapping:
    voxel_size_lr_mm: float | None
    voxel_size_ap_mm: float | None
    voxel_size_is_mm: float | None
    axis_mapping_successful: bool
    orientation_codes: tuple[str, str, str] | None
    qc_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class NiftiHeaderRecord:
    header_type: Literal["NIfTI-1", "NIfTI-2"]
    image_shape: tuple[int, ...]
    ndim: int
    n_volumes: int | None
    native_voxel_size_x_mm: float
    native_voxel_size_y_mm: float
    native_voxel_size_z_mm: float
    qform_code: int
    sform_code: int
    best_affine: tuple[tuple[float, ...], ...]
    orientation_codes: tuple[str, str, str] | None
    voxel_size_lr_mm: float | None
    voxel_size_ap_mm: float | None
    voxel_size_is_mm: float | None
    axis_mapping_successful: bool
    canonical_voxel_size_lr_mm: float | None
    canonical_voxel_size_ap_mm: float | None
    canonical_voxel_size_is_mm: float | None
    qc_flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def native_zooms(self) -> tuple[float, float, float]:
        return (
            self.native_voxel_size_x_mm,
            self.native_voxel_size_y_mm,
            self.native_voxel_size_z_mm,
        )


def decompress_gzip_prefix(
    compressed: bytes,
    *,
    min_uncompressed: int = MIN_UNCOMPRESSED_HEADER_BYTES,
) -> bytes:
    """Decompress as much of a partial gzip stream as needed for the header."""

    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        decompressed = decompressor.decompress(compressed, min_uncompressed)
    except zlib.error as exc:
        raise HeaderParseError(f"Malformed gzip stream: {exc}") from exc
    if len(decompressed) < min_uncompressed and not decompressor.eof:
        raise InsufficientHeaderBytes(
            f"Only decompressed {len(decompressed)} bytes; need {min_uncompressed}"
        )
    return decompressed


def parse_header_from_object_bytes(object_bytes: bytes, *, is_gzipped: bool) -> NiftiHeaderRecord:
    """Parse a NIfTI header from ranged object bytes."""

    header_bytes = decompress_gzip_prefix(object_bytes) if is_gzipped else object_bytes
    return parse_nifti_header_bytes(header_bytes)


def parse_nifti_header_bytes(header_bytes: bytes) -> NiftiHeaderRecord:
    """Parse NIfTI-1 or NIfTI-2 headers using NiBabel header classes."""

    header_type, header_size, header_cls = _detect_header_type(header_bytes)
    if len(header_bytes) < header_size:
        raise InsufficientHeaderBytes(
            f"Only have {len(header_bytes)} bytes; need {header_size} for {header_type}"
        )

    try:
        header = header_cls.from_fileobj(io.BytesIO(header_bytes[:header_size]))
    except Exception as exc:  # nibabel raises several domain-specific exceptions here
        raise HeaderParseError(f"Could not parse {header_type} header: {exc}") from exc

    return header_to_record(header, header_type=header_type)


def header_to_record(
    header: nib.nifti1.Nifti1Header | nib.nifti2.Nifti2Header,
    *,
    header_type: Literal["NIfTI-1", "NIfTI-2"] | None = None,
) -> NiftiHeaderRecord:
    """Convert a NiBabel header to a serializable scan record."""

    detected_type = header_type or (
        "NIfTI-2" if isinstance(header, nib.nifti2.Nifti2Header) else "NIfTI-1"
    )
    shape = tuple(int(value) for value in header.get_data_shape())
    ndim = len(shape)
    n_volumes = int(shape[3]) if ndim >= 4 else None

    zoom_values = tuple(float(value) for value in header.get_zooms()[:3])
    if len(zoom_values) < 3:
        raise HeaderParseError(f"Header has fewer than three spatial zooms: {zoom_values}")
    zooms = cast(tuple[float, float, float], zoom_values)

    qform_code = int(header["qform_code"])
    sform_code = int(header["sform_code"])
    affine = np.array(cast(Any, header.get_best_affine()), dtype=np.float64)
    mapping = map_voxel_sizes_to_ras(zooms, affine)

    qc_flags = list(mapping.qc_flags)
    if any(not math.isfinite(value) or value <= 0 for value in zooms):
        qc_flags.append("invalid_native_voxel_size")
    if any(value < 0.25 or value > 20 for value in zooms if math.isfinite(value)):
        qc_flags.append("implausible_native_voxel_size")

    canonical = tuple(
        round(value, 3) if value is not None and math.isfinite(value) else None
        for value in (
            mapping.voxel_size_lr_mm,
            mapping.voxel_size_ap_mm,
            mapping.voxel_size_is_mm,
        )
    )

    return NiftiHeaderRecord(
        header_type=detected_type,
        image_shape=shape,
        ndim=ndim,
        n_volumes=n_volumes,
        native_voxel_size_x_mm=zooms[0],
        native_voxel_size_y_mm=zooms[1],
        native_voxel_size_z_mm=zooms[2],
        qform_code=qform_code,
        sform_code=sform_code,
        best_affine=tuple(tuple(float(value) for value in row) for row in affine),
        orientation_codes=mapping.orientation_codes,
        voxel_size_lr_mm=mapping.voxel_size_lr_mm,
        voxel_size_ap_mm=mapping.voxel_size_ap_mm,
        voxel_size_is_mm=mapping.voxel_size_is_mm,
        axis_mapping_successful=mapping.axis_mapping_successful,
        canonical_voxel_size_lr_mm=canonical[0],
        canonical_voxel_size_ap_mm=canonical[1],
        canonical_voxel_size_is_mm=canonical[2],
        qc_flags=tuple(dict.fromkeys(qc_flags)),
    )


def map_voxel_sizes_to_ras(
    native_zooms: tuple[float, float, float], affine: np.ndarray
) -> OrientationMapping:
    """Map native voxel dimensions to RAS anatomical axes using an affine."""

    qc_flags: list[str] = []
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        return OrientationMapping(None, None, None, False, None, ("invalid_affine",))

    spatial = affine[:3, :3]
    if np.linalg.matrix_rank(spatial) < 3 or abs(float(np.linalg.det(spatial))) < 1e-12:
        return OrientationMapping(None, None, None, False, None, ("invalid_affine",))

    column_norms = np.linalg.norm(spatial, axis=0)
    if not np.allclose(column_norms, native_zooms, rtol=1e-3, atol=1e-4):
        qc_flags.append("affine_zoom_mismatch")

    try:
        orientation = io_orientation(affine)
        axis_codes = nib.aff2axcodes(affine)
    except Exception:
        return OrientationMapping(None, None, None, False, None, ("orientation_mapping_failed",))

    world_axes: list[int] = []
    for voxel_axis in range(3):
        axis = orientation[voxel_axis, 0]
        if np.isnan(axis):
            return OrientationMapping(
                None, None, None, False, None, ("orientation_mapping_failed",)
            )
        world_axes.append(int(axis))

    if sorted(world_axes) != [0, 1, 2]:
        return OrientationMapping(None, None, None, False, None, ("orientation_mapping_failed",))

    ras_sizes: dict[int, float] = {}
    for voxel_axis, world_axis in enumerate(world_axes):
        ras_sizes[world_axis] = float(native_zooms[voxel_axis])

    return OrientationMapping(
        voxel_size_lr_mm=ras_sizes[0],
        voxel_size_ap_mm=ras_sizes[1],
        voxel_size_is_mm=ras_sizes[2],
        axis_mapping_successful=True,
        orientation_codes=cast(tuple[str, str, str], tuple(str(code) for code in axis_codes)),
        qc_flags=tuple(qc_flags),
    )


def _detect_header_type(
    header_bytes: bytes,
) -> tuple[
    Literal["NIfTI-1", "NIfTI-2"],
    int,
    type[nib.nifti1.Nifti1Header] | type[nib.nifti2.Nifti2Header],
]:
    if len(header_bytes) < 4:
        raise InsufficientHeaderBytes("Need at least four bytes to detect NIfTI header type")

    little = struct.unpack("<i", header_bytes[:4])[0]
    big = struct.unpack(">i", header_bytes[:4])[0]
    if little == NIFTI1_HEADER_SIZE or big == NIFTI1_HEADER_SIZE:
        return "NIfTI-1", NIFTI1_HEADER_SIZE, nib.nifti1.Nifti1Header
    if little == NIFTI2_HEADER_SIZE or big == NIFTI2_HEADER_SIZE:
        return "NIfTI-2", NIFTI2_HEADER_SIZE, nib.nifti2.Nifti2Header
    raise HeaderParseError(f"Unrecognized NIfTI sizeof_hdr value: little={little}, big={big}")
