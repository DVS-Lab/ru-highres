"""JSON scan summaries and CSV error exports."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from openneuro_voxels.database import utc_now


def write_reports(*, database: Path, results_dir: Path = Path("results")) -> dict[str, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "scan_summary.json"
    errors_path = results_dir / "file_level_errors.csv"
    summary = build_scan_summary(database)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    export_errors(database, errors_path)
    return {"scan_summary": summary_path, "file_level_errors": errors_path}


def build_scan_summary(database: Path) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        datasets_examined = _scalar(connection, "SELECT COUNT(*) FROM datasets")
        datasets_with_bold = _scalar(
            connection, "SELECT COUNT(*) FROM datasets WHERE has_bold = 1"
        )
        bold_files = _scalar(connection, "SELECT COUNT(*) FROM s3_objects")
        parsed = _scalar(
            connection, "SELECT COUNT(*) FROM s3_objects WHERE scan_status = 'success'"
        )
        failed = _scalar(connection, "SELECT COUNT(*) FROM s3_objects WHERE scan_status = 'failed'")
        bytes_retrieved = _scalar(
            connection,
            "SELECT COALESCE(SUM(compressed_bytes_retrieved), 0) FROM s3_objects",
        )
        rows = connection.execute(
            """
            SELECT qc_flags
            FROM nifti_headers
            """
        ).fetchall()
        qc_counts: Counter[str] = Counter()
        for row in rows:
            for flag in json.loads(row["qc_flags"]):
                qc_counts[flag] += 1
        resolution_rows = connection.execute(
            """
            SELECT
                o.dataset_accession,
                h.canonical_voxel_size_lr_mm,
                h.canonical_voxel_size_ap_mm,
                h.canonical_voxel_size_is_mm
            FROM s3_objects o
            JOIN nifti_headers h ON h.key = o.key
            WHERE h.axis_mapping_successful = 1
            GROUP BY
                o.dataset_accession,
                h.canonical_voxel_size_lr_mm,
                h.canonical_voxel_size_ap_mm,
                h.canonical_voxel_size_is_mm
            """
        ).fetchall()
    finally:
        connection.close()

    per_dataset = Counter(row["dataset_accession"] for row in resolution_rows)
    return {
        "scan_summary_created_at": utc_now(),
        "openneuro_bucket_snapshot": "current public S3 object tree at scan time",
        "number_openneuro_dataset_prefixes_examined": datasets_examined,
        "number_containing_bold_data": datasets_with_bold,
        "number_bold_files_discovered": bold_files,
        "number_successfully_parsed": parsed,
        "number_failed": failed,
        "failure_percentage": (100.0 * failed / bold_files) if bold_files else 0.0,
        "total_compressed_bytes_retrieved": bytes_retrieved,
        "number_unique_dataset_resolution_observations": len(resolution_rows),
        "number_datasets_with_one_resolution": sum(
            1 for count in per_dataset.values() if count == 1
        ),
        "number_datasets_with_multiple_resolutions": sum(
            1 for count in per_dataset.values() if count > 1
        ),
        "qc_flag_counts": dict(sorted(qc_counts.items())),
    }


def export_errors(database: Path, output_csv: Path) -> None:
    connection = sqlite3.connect(database)
    try:
        frame = pd.read_sql_query(
            """
            SELECT
                key,
                dataset_accession,
                error_class,
                error_message,
                compressed_bytes_retrieved,
                retry_count,
                created_at
            FROM scan_errors
            ORDER BY dataset_accession, key, created_at
            """,
            connection,
        )
    finally:
        connection.close()
    frame.to_csv(output_csv, index=False)


def _scalar(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])
