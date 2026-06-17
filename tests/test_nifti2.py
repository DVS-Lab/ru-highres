from __future__ import annotations

from openneuro_voxels.nifti_header import parse_nifti_header_bytes


def test_parse_nifti2(nifti2_header_factory) -> None:
    record = parse_nifti_header_bytes(nifti2_header_factory())

    assert record.header_type == "NIfTI-2"
    assert record.image_shape == (3, 4, 5)
    assert record.ndim == 3
    assert record.n_volumes is None
    assert record.native_zooms == (1.5, 2.0, 2.5)
