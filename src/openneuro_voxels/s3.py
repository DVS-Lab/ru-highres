"""Anonymous OpenNeuro S3 access helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import cast

import boto3
from botocore import UNSIGNED
from botocore.client import BaseClient, Config

from openneuro_voxels.bids_entities import BidsEntities, is_raw_bold_key, parse_bids_entities

DATASET_PREFIX_RE = re.compile(r"^ds[0-9A-Za-z]+/$")


@dataclass(frozen=True)
class S3Object:
    dataset: str
    key: str
    size: int
    etag: str
    last_modified: str
    entities: BidsEntities


def anonymous_openneuro_client(
    *, region_name: str = "us-east-1", timeout: int = 30
) -> BaseClient:
    """Create an unsigned S3 client for the public OpenNeuro bucket."""

    config = Config(
        signature_version=UNSIGNED,
        region_name=region_name,
        connect_timeout=timeout,
        read_timeout=timeout,
        retries={"max_attempts": 1},
    )
    return boto3.client("s3", config=config, region_name=region_name)


def list_dataset_prefixes(client: BaseClient, *, bucket: str = "openneuro.org") -> Iterator[str]:
    """Yield top-level OpenNeuro dataset prefixes."""

    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Delimiter="/"):
        for prefix in page.get("CommonPrefixes", []):
            value = prefix.get("Prefix", "")
            if DATASET_PREFIX_RE.match(value):
                yield value.rstrip("/")


def list_bold_objects(
    client: BaseClient,
    dataset_accession: str,
    *,
    bucket: str = "openneuro.org",
) -> Iterator[S3Object]:
    """Yield raw BOLD NIfTI objects for one dataset accession."""

    paginator = client.get_paginator("list_objects_v2")
    prefix = f"{dataset_accession}/"
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not is_raw_bold_key(key):
                continue
            yield S3Object(
                dataset=dataset_accession,
                key=key,
                size=int(item["Size"]),
                etag=str(item["ETag"]).strip('"'),
                last_modified=item["LastModified"].isoformat(),
                entities=parse_bids_entities(key),
            )


def get_range(
    client: BaseClient,
    *,
    bucket: str,
    key: str,
    range_bytes: int,
) -> bytes:
    """Fetch bytes from the beginning of an S3 object without a full GET."""

    response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{range_bytes - 1}")
    return cast(bytes, response["Body"].read())


def parse_dataset_file(path: str) -> list[str]:
    """Read OpenNeuro accessions from a plain text file."""

    datasets: list[str] = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            datasets.append(value)
    return datasets


def unique_datasets(values: Iterable[str]) -> list[str]:
    """Preserve input order while removing duplicate accessions."""

    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique
