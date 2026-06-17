"""Discovery and partial-header scan orchestration."""

from __future__ import annotations

import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from botocore.client import BaseClient
from botocore.exceptions import BotoCoreError, ClientError

from openneuro_voxels import __version__
from openneuro_voxels.config import ScanConfig
from openneuro_voxels.database import VoxelScanDatabase
from openneuro_voxels.nifti_header import (
    HeaderParseError,
    InsufficientHeaderBytes,
    parse_header_from_object_bytes,
)
from openneuro_voxels.s3 import (
    S3Object,
    anonymous_openneuro_client,
    get_range,
    list_bold_objects,
    list_dataset_prefixes,
)
from openneuro_voxels.sidecars import classify_acquisition_metadata, load_acquisition_metadata

DEFAULT_RANGE_STEPS = (8 * 1024, 32 * 1024, 128 * 1024, 512 * 1024, 2 * 1024 * 1024)


@dataclass(frozen=True)
class ScanObjectResult:
    key: str
    dataset: str
    ok: bool
    compressed_bytes_retrieved: int
    retry_count: int
    error_class: str | None = None
    error_message: str | None = None


class RateLimiter:
    """Thread-safe fixed-window-ish limiter for polite S3 access."""

    def __init__(self, requests_per_second: float | None) -> None:
        self.interval = 1.0 / requests_per_second if requests_per_second else 0.0
        self._lock = Lock()
        self._next_time = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                time.sleep(self._next_time - now)
                now = time.monotonic()
            self._next_time = now + self.interval


def discover(
    *,
    database_path: Path,
    client: BaseClient,
    bucket: str,
    datasets: list[str] | None = None,
    max_datasets: int | None = None,
) -> dict[str, int]:
    """Populate the object manifest in SQLite."""

    db = VoxelScanDatabase(database_path)
    db.initialize()
    try:
        accessions = datasets if datasets else list(list_dataset_prefixes(client, bucket=bucket))
        if max_datasets is not None:
            accessions = accessions[:max_datasets]

        total_objects = 0
        for accession in accessions:
            count = 0
            with db.transaction():
                db.add_dataset(accession)
                for obj in list_bold_objects(client, accession, bucket=bucket):
                    db.upsert_object(obj)
                    count += 1
                    total_objects += 1
                db.mark_dataset_discovery_complete(accession, count)
        return {"datasets_examined": len(accessions), "bold_objects": total_objects}
    finally:
        db.close()


def scan_database(
    *,
    config: ScanConfig,
    client: BaseClient | None = None,
    datasets: list[str] | None = None,
) -> dict[str, int]:
    """Scan discovered objects in the database using bounded concurrency."""

    db = VoxelScanDatabase(config.database)
    db.initialize()
    run_id = db.begin_scan_run(
        command="scan",
        software_version=__version__,
        git_commit=get_git_commit(),
        parameters={
            "workers": config.workers,
            "max_range_bytes": config.max_range_bytes,
            "retries": config.retries,
            "timeout": config.timeout,
            "fallback": config.fallback,
            "datasets": datasets,
            "force": config.force,
        },
    )
    db.close()

    s3_client = client or anonymous_openneuro_client(
        region_name=config.region, timeout=config.timeout
    )
    rows = _load_rows_to_scan(config.database, datasets=datasets, force=config.force)
    limiter = RateLimiter(config.requests_per_second)

    successes = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = [
            executor.submit(
                scan_one_object,
                client=s3_client,
                bucket=config.bucket,
                obj=_row_to_s3_object(row),
                config=config,
                limiter=limiter,
            )
            for row in rows
        ]
        for future in as_completed(futures):
            result = future.result()
            if result.ok:
                successes += 1
            else:
                failures += 1

    db = VoxelScanDatabase(config.database)
    db.initialize()
    try:
        db.finish_scan_run(run_id)
    finally:
        db.close()
    return {"requested": len(rows), "successes": successes, "failures": failures}


def scan_one_object(
    *,
    client: BaseClient,
    bucket: str,
    obj: S3Object,
    config: ScanConfig,
    limiter: RateLimiter | None = None,
) -> ScanObjectResult:
    """Retrieve and parse a single NIfTI header, recording the result."""

    db = VoxelScanDatabase(config.database)
    db.initialize()
    range_steps = tuple(step for step in DEFAULT_RANGE_STEPS if step <= config.max_range_bytes)
    if not range_steps or range_steps[-1] != config.max_range_bytes:
        range_steps = (*range_steps, config.max_range_bytes)

    compressed_bytes = 0
    retry_count = 0
    try:
        if db.should_skip_success(obj.key, obj.etag, force=config.force):
            return ScanObjectResult(obj.key, obj.dataset, True, 0, 0)

        last_error: Exception | None = None
        for range_bytes in range_steps:
            for attempt in range(config.retries + 1):
                retry_count += int(attempt > 0)
                try:
                    if limiter is not None:
                        limiter.wait()
                    payload = get_range(client, bucket=bucket, key=obj.key, range_bytes=range_bytes)
                    compressed_bytes = max(compressed_bytes, len(payload))
                    record = parse_header_from_object_bytes(
                        payload, is_gzipped=obj.key.endswith(".nii.gz")
                    )
                    try:
                        metadata = load_acquisition_metadata(
                            client,
                            bucket=bucket,
                            key=obj.key,
                            entities=obj.entities,
                        )
                    except (BotoCoreError, ClientError, OSError):
                        metadata = classify_acquisition_metadata(
                            {},
                            metadata_status="error",
                            source_keys=(),
                        )
                except InsufficientHeaderBytes as exc:
                    last_error = exc
                    break
                except (BotoCoreError, ClientError, HeaderParseError) as exc:
                    last_error = exc
                    if attempt >= config.retries:
                        break
                    _sleep_for_retry(attempt)
                    continue
                else:
                    with db.transaction():
                        db.record_acquisition_metadata(key=obj.key, metadata=metadata)
                        db.record_header(
                            key=obj.key,
                            record=record,
                            compressed_bytes_retrieved=compressed_bytes,
                            retry_count=retry_count,
                            software_version=__version__,
                            git_commit=get_git_commit(),
                        )
                    return ScanObjectResult(
                        obj.key,
                        obj.dataset,
                        True,
                        compressed_bytes,
                        retry_count,
                    )

        message = (
            f"Header unavailable within {config.max_range_bytes} bytes"
            if isinstance(last_error, InsufficientHeaderBytes)
            else str(last_error)
        )
        with db.transaction():
            db.record_error(
                key=obj.key,
                dataset_accession=obj.dataset,
                error_class=type(last_error).__name__ if last_error else "UnknownError",
                error_message=message,
                compressed_bytes_retrieved=compressed_bytes,
                retry_count=retry_count,
            )
        return ScanObjectResult(
            obj.key,
            obj.dataset,
            False,
            compressed_bytes,
            retry_count,
            type(last_error).__name__ if last_error else "UnknownError",
            message,
        )
    finally:
        db.close()


def discover_and_scan(
    *,
    config: ScanConfig,
    datasets: list[str] | None = None,
    max_datasets: int | None = None,
) -> dict[str, int]:
    client = anonymous_openneuro_client(region_name=config.region, timeout=config.timeout)
    discovery = discover(
        database_path=config.database,
        client=client,
        bucket=config.bucket,
        datasets=datasets,
        max_datasets=max_datasets,
    )
    scanned = scan_database(config=config, client=client, datasets=datasets)
    return {**discovery, **scanned}


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return f"{commit}-dirty" if status.stdout.strip() else commit


def _load_rows_to_scan(
    database_path: Path, *, datasets: list[str] | None, force: bool
) -> list[Any]:
    db = VoxelScanDatabase(database_path)
    db.initialize()
    try:
        return db.objects_to_scan(datasets=datasets, force=force)
    finally:
        db.close()


def _row_to_s3_object(row: Any) -> S3Object:
    return S3Object(
        dataset=row["dataset_accession"],
        key=row["key"],
        size=int(row["size"]),
        etag=row["etag"],
        last_modified=row["last_modified"],
        entities=_entities_from_row(row),
    )


def _entities_from_row(row: Any) -> Any:
    from openneuro_voxels.bids_entities import BidsEntities

    return BidsEntities(
        subject=row["subject"],
        session=row["session"],
        task=row["task"],
        acquisition=row["acquisition"],
        reconstruction=row["reconstruction"],
        direction=row["direction"],
        run=row["run"],
        echo=row["echo"],
        part=row["part"],
    )


def _sleep_for_retry(attempt: int) -> None:
    base = min(2**attempt, 30)
    time.sleep(base + random.uniform(0, 0.25 * base))
