from __future__ import annotations

import numpy as np

from openneuro_voxels.bids_entities import parse_bids_entities
from openneuro_voxels.database import VoxelScanDatabase
from openneuro_voxels.nifti_header import header_to_record
from openneuro_voxels.s3 import S3Object


def test_successful_same_etag_is_skipped(tmp_path) -> None:
    database = tmp_path / "scan.sqlite"
    db = VoxelScanDatabase(database)
    db.initialize()
    key = "ds1/sub-01/func/sub-01_task-rest_bold.nii.gz"
    obj = S3Object(
        dataset="ds1",
        key=key,
        size=123,
        etag="abc",
        last_modified="2026-06-16T00:00:00+00:00",
        entities=parse_bids_entities(key),
    )
    with db.transaction():
        db.upsert_object(obj)
        db.record_header(
            key=key,
            record=_record(),
            compressed_bytes_retrieved=8192,
            retry_count=0,
            software_version="test",
            git_commit="deadbeef",
        )

    assert db.should_skip_success(key, "abc")
    assert not db.should_skip_success(key, "changed")
    assert not db.should_skip_success(key, "abc", force=True)
    db.close()


def _record():
    import nibabel as nib

    header = nib.Nifti1Header()
    header.set_data_shape((2, 2, 2))
    header.set_zooms((2.0, 2.0, 2.0))
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    header.set_sform(affine, code=1)
    header.set_qform(affine, code=1)
    return header_to_record(header)
