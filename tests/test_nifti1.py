from __future__ import annotations

import pytest

from openneuro_voxels.nifti_header import parse_nifti_header_bytes


def test_parse_nifti1_little_endian(nifti1_header_factory) -> None:
    record = parse_nifti_header_bytes(nifti1_header_factory())

    assert record.header_type == "NIfTI-1"
    assert record.image_shape == (4, 5, 6, 7)
    assert record.n_volumes == 7
    assert record.native_zooms == pytest.approx((2.4, 2.4, 3.0))
    assert record.canonical_voxel_size_lr_mm == 2.4


def test_parse_nifti1_big_endian(nifti1_header_factory) -> None:
    record = parse_nifti_header_bytes(nifti1_header_factory(endian=">"))

    assert record.header_type == "NIfTI-1"
    assert record.native_zooms == pytest.approx((2.4, 2.4, 3.0))
    assert record.axis_mapping_successful
