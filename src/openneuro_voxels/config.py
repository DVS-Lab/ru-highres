"""Configuration dataclasses for scanner commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScanConfig:
    database: Path = Path("results/openneuro_voxel_scan.sqlite")
    bucket: str = "openneuro.org"
    region: str = "us-east-1"
    workers: int = 16
    requests_per_second: float | None = None
    max_range_bytes: int = 2 * 1024 * 1024
    retries: int = 3
    timeout: int = 30
    resume: bool = True
    force: bool = False
    fallback: str = "none"
    log_level: str = "INFO"


@dataclass(frozen=True)
class OutputConfig:
    results_dir: Path = Path("results")
    figures_dir: Path = Path("figures")
    round_decimals: int = 3
    sensitivity_min_files: int = 2
    sensitivity_min_dataset_percent: float = 5.0
