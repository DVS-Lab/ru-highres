"""Explicit full-file validation for selected objects."""

from __future__ import annotations

import random
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import pandas as pd
from botocore.client import BaseClient

from openneuro_voxels.nifti_header import parse_header_from_object_bytes
from openneuro_voxels.s3 import get_range


def sample_successful_keys(*, database: Path, sample_size: int, seed: int) -> list[str]:
    """Sample successfully parsed keys reproducibly from a scan database."""

    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            """
            SELECT key
            FROM s3_objects
            WHERE scan_status = 'success'
            ORDER BY key
            """
        ).fetchall()
    finally:
        connection.close()

    keys = [str(row[0]) for row in rows]
    if not keys:
        return []
    rng = random.Random(seed)
    return rng.sample(keys, k=min(sample_size, len(keys)))


def validate_sample(
    *,
    keys: list[str],
    client: BaseClient,
    bucket: str = "openneuro.org",
    max_range_bytes: int = 2 * 1024 * 1024,
) -> pd.DataFrame:
    """Download explicitly selected full files and compare header zooms."""

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for key in keys:
            partial = get_range(client, bucket=bucket, key=key, range_bytes=max_range_bytes)
            partial_header = parse_header_from_object_bytes(
                partial, is_gzipped=key.endswith(".nii.gz")
            )

            suffix = ".nii.gz" if key.endswith(".nii.gz") else ".nii"
            local_path = Path(tmpdir) / f"validation{suffix}"
            response = client.get_object(Bucket=bucket, Key=key)
            local_path.write_bytes(response["Body"].read())
            full = nib.load(str(local_path))
            full_header = cast(Any, full.header)
            full_zooms = tuple(float(value) for value in full_header.get_zooms()[:3])
            partial_zooms = partial_header.native_zooms
            diffs = [abs(a - b) for a, b in zip(partial_zooms, full_zooms, strict=True)]
            rows.append(
                {
                    "key": key,
                    "partial_native_zooms": partial_zooms,
                    "full_native_zooms": full_zooms,
                    "max_abs_difference_mm": max(diffs),
                    "passes_1e_6_mm": max(diffs) <= 1e-6,
                }
            )
    return pd.DataFrame(rows)
