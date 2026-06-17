from __future__ import annotations

import numpy as np

from openneuro_voxels.nifti_header import parse_nifti_header_bytes


def test_axis_permutation_maps_voxel_sizes_to_ras(nifti1_header_factory) -> None:
    affine = np.array(
        [
            [0.0, 0.0, 4.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    header = nifti1_header_factory(shape=(4, 5, 6), zooms=(2.0, 3.0, 4.0), affine=affine)

    record = parse_nifti_header_bytes(header)

    assert record.axis_mapping_successful
    assert record.voxel_size_lr_mm == 4.0
    assert record.voxel_size_ap_mm == 2.0
    assert record.voxel_size_is_mm == 3.0


def test_invalid_affine_is_flagged(nifti1_header_factory) -> None:
    affine = np.zeros((4, 4))
    affine[3, 3] = 1.0
    header = nifti1_header_factory(shape=(4, 5, 6), zooms=(2.0, 2.0, 2.0), affine=affine)

    record = parse_nifti_header_bytes(header)

    assert not record.axis_mapping_successful
    assert "invalid_affine" in record.qc_flags
