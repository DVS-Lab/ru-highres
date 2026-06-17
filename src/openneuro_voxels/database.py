"""SQLite state and audit store for OpenNeuro voxel scans."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openneuro_voxels.nifti_header import NiftiHeaderRecord
from openneuro_voxels.s3 import S3Object

SCHEMA_VERSION = 1


class VoxelScanDatabase:
    """Small SQLite wrapper with explicit schema and resumability helpers."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        self.connection.close()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS datasets (
                accession TEXT PRIMARY KEY,
                discovered_at TEXT NOT NULL,
                has_bold INTEGER NOT NULL DEFAULT 0,
                object_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS s3_objects (
                key TEXT PRIMARY KEY,
                dataset_accession TEXT NOT NULL,
                size INTEGER NOT NULL,
                etag TEXT NOT NULL,
                last_modified TEXT NOT NULL,
                subject TEXT,
                session TEXT,
                task TEXT,
                acquisition TEXT,
                reconstruction TEXT,
                direction TEXT,
                run TEXT,
                echo TEXT,
                part TEXT,
                scan_status TEXT NOT NULL DEFAULT 'discovered',
                compressed_bytes_retrieved INTEGER,
                retry_count INTEGER NOT NULL DEFAULT 0,
                extraction_timestamp TEXT,
                software_version TEXT,
                git_commit TEXT,
                FOREIGN KEY(dataset_accession) REFERENCES datasets(accession)
            );

            CREATE TABLE IF NOT EXISTS nifti_headers (
                key TEXT PRIMARY KEY,
                header_type TEXT NOT NULL,
                image_shape TEXT NOT NULL,
                ndim INTEGER NOT NULL,
                n_volumes INTEGER,
                native_voxel_size_x_mm REAL NOT NULL,
                native_voxel_size_y_mm REAL NOT NULL,
                native_voxel_size_z_mm REAL NOT NULL,
                qform_code INTEGER NOT NULL,
                sform_code INTEGER NOT NULL,
                best_affine TEXT NOT NULL,
                orientation_codes TEXT,
                voxel_size_lr_mm REAL,
                voxel_size_ap_mm REAL,
                voxel_size_is_mm REAL,
                axis_mapping_successful INTEGER NOT NULL,
                canonical_voxel_size_lr_mm REAL,
                canonical_voxel_size_ap_mm REAL,
                canonical_voxel_size_is_mm REAL,
                qc_flags TEXT NOT NULL,
                FOREIGN KEY(key) REFERENCES s3_objects(key) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS scan_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                dataset_accession TEXT NOT NULL,
                error_class TEXT NOT NULL,
                error_message TEXT NOT NULL,
                compressed_bytes_retrieved INTEGER,
                retry_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(key) REFERENCES s3_objects(key) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS scan_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                command TEXT NOT NULL,
                software_version TEXT NOT NULL,
                git_commit TEXT,
                parameters TEXT NOT NULL
            );
            """
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self.connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        try:
            yield
        except Exception:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def add_dataset(self, accession: str) -> None:
        self.connection.execute(
            """
            INSERT INTO datasets(accession, discovered_at)
            VALUES (?, ?)
            ON CONFLICT(accession) DO NOTHING
            """,
            (accession, utc_now()),
        )

    def upsert_object(self, obj: S3Object) -> None:
        self.add_dataset(obj.dataset)
        entity_values = obj.entities.as_dict()
        self.connection.execute(
            """
            INSERT INTO s3_objects(
                key, dataset_accession, size, etag, last_modified,
                subject, session, task, acquisition, reconstruction, direction,
                run, echo, part, scan_status
            )
            VALUES (
                :key, :dataset_accession, :size, :etag, :last_modified,
                :subject, :session, :task, :acquisition, :reconstruction, :direction,
                :run, :echo, :part, 'discovered'
            )
            ON CONFLICT(key) DO UPDATE SET
                dataset_accession=excluded.dataset_accession,
                size=excluded.size,
                etag=excluded.etag,
                last_modified=excluded.last_modified,
                subject=excluded.subject,
                session=excluded.session,
                task=excluded.task,
                acquisition=excluded.acquisition,
                reconstruction=excluded.reconstruction,
                direction=excluded.direction,
                run=excluded.run,
                echo=excluded.echo,
                part=excluded.part
            """,
            {
                "key": obj.key,
                "dataset_accession": obj.dataset,
                "size": obj.size,
                "etag": obj.etag,
                "last_modified": obj.last_modified,
                **entity_values,
            },
        )

    def mark_dataset_discovery_complete(self, accession: str, object_count: int) -> None:
        self.connection.execute(
            """
            UPDATE datasets
            SET has_bold = ?, object_count = ?
            WHERE accession = ?
            """,
            (int(object_count > 0), object_count, accession),
        )

    def should_skip_success(self, key: str, etag: str, *, force: bool = False) -> bool:
        if force:
            return False
        row = self.connection.execute(
            """
            SELECT scan_status, etag
            FROM s3_objects
            WHERE key = ?
            """,
            (key,),
        ).fetchone()
        return bool(row and row["scan_status"] == "success" and row["etag"] == etag)

    def objects_to_scan(
        self, *, datasets: list[str] | None = None, force: bool = False
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[Any] = []
        if datasets:
            placeholders = ",".join("?" for _ in datasets)
            clauses.append(f"dataset_accession IN ({placeholders})")
            params.extend(datasets)
        if not force:
            clauses.append("scan_status != 'success'")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return list(
            self.connection.execute(
                f"""
                SELECT *
                FROM s3_objects
                {where}
                ORDER BY dataset_accession, key
                """,
                params,
            )
        )

    def record_header(
        self,
        *,
        key: str,
        record: NiftiHeaderRecord,
        compressed_bytes_retrieved: int,
        retry_count: int,
        software_version: str,
        git_commit: str | None,
    ) -> None:
        payload = asdict(record)
        self.connection.execute(
            """
            INSERT INTO nifti_headers(
                key, header_type, image_shape, ndim, n_volumes,
                native_voxel_size_x_mm, native_voxel_size_y_mm, native_voxel_size_z_mm,
                qform_code, sform_code, best_affine, orientation_codes,
                voxel_size_lr_mm, voxel_size_ap_mm, voxel_size_is_mm,
                axis_mapping_successful,
                canonical_voxel_size_lr_mm, canonical_voxel_size_ap_mm,
                canonical_voxel_size_is_mm, qc_flags
            )
            VALUES (
                :key, :header_type, :image_shape, :ndim, :n_volumes,
                :native_voxel_size_x_mm, :native_voxel_size_y_mm, :native_voxel_size_z_mm,
                :qform_code, :sform_code, :best_affine, :orientation_codes,
                :voxel_size_lr_mm, :voxel_size_ap_mm, :voxel_size_is_mm,
                :axis_mapping_successful,
                :canonical_voxel_size_lr_mm, :canonical_voxel_size_ap_mm,
                :canonical_voxel_size_is_mm, :qc_flags
            )
            ON CONFLICT(key) DO UPDATE SET
                header_type=excluded.header_type,
                image_shape=excluded.image_shape,
                ndim=excluded.ndim,
                n_volumes=excluded.n_volumes,
                native_voxel_size_x_mm=excluded.native_voxel_size_x_mm,
                native_voxel_size_y_mm=excluded.native_voxel_size_y_mm,
                native_voxel_size_z_mm=excluded.native_voxel_size_z_mm,
                qform_code=excluded.qform_code,
                sform_code=excluded.sform_code,
                best_affine=excluded.best_affine,
                orientation_codes=excluded.orientation_codes,
                voxel_size_lr_mm=excluded.voxel_size_lr_mm,
                voxel_size_ap_mm=excluded.voxel_size_ap_mm,
                voxel_size_is_mm=excluded.voxel_size_is_mm,
                axis_mapping_successful=excluded.axis_mapping_successful,
                canonical_voxel_size_lr_mm=excluded.canonical_voxel_size_lr_mm,
                canonical_voxel_size_ap_mm=excluded.canonical_voxel_size_ap_mm,
                canonical_voxel_size_is_mm=excluded.canonical_voxel_size_is_mm,
                qc_flags=excluded.qc_flags
            """,
            {
                **payload,
                "key": key,
                "image_shape": json.dumps(payload["image_shape"]),
                "best_affine": json.dumps(payload["best_affine"]),
                "orientation_codes": json.dumps(payload["orientation_codes"])
                if payload["orientation_codes"] is not None
                else None,
                "axis_mapping_successful": int(payload["axis_mapping_successful"]),
                "qc_flags": json.dumps(payload["qc_flags"]),
            },
        )
        self.connection.execute(
            """
            UPDATE s3_objects
            SET scan_status='success',
                compressed_bytes_retrieved=?,
                retry_count=?,
                extraction_timestamp=?,
                software_version=?,
                git_commit=?
            WHERE key=?
            """,
            (
                compressed_bytes_retrieved,
                retry_count,
                utc_now(),
                software_version,
                git_commit,
                key,
            ),
        )

    def record_error(
        self,
        *,
        key: str,
        dataset_accession: str,
        error_class: str,
        error_message: str,
        compressed_bytes_retrieved: int,
        retry_count: int,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO scan_errors(
                key, dataset_accession, error_class, error_message,
                compressed_bytes_retrieved, retry_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                dataset_accession,
                error_class,
                error_message[:2000],
                compressed_bytes_retrieved,
                retry_count,
                utc_now(),
            ),
        )
        self.connection.execute(
            """
            UPDATE s3_objects
            SET scan_status='failed',
                compressed_bytes_retrieved=?,
                retry_count=?,
                extraction_timestamp=?
            WHERE key=?
            """,
            (compressed_bytes_retrieved, retry_count, utc_now(), key),
        )

    def begin_scan_run(
        self,
        *,
        command: str,
        software_version: str,
        git_commit: str | None,
        parameters: dict[str, Any],
    ) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO scan_runs(started_at, command, software_version, git_commit, parameters)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                command,
                software_version,
                git_commit,
                json.dumps(parameters, sort_keys=True),
            ),
        )
        self.connection.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return a scan run id")
        return int(cursor.lastrowid)

    def finish_scan_run(self, run_id: int) -> None:
        self.connection.execute(
            "UPDATE scan_runs SET finished_at = ? WHERE id = ?",
            (utc_now(), run_id),
        )
        self.connection.commit()


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
