"""Command-line interface for OpenNeuro voxel-resolution scans."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from openneuro_voxels.aggregate import write_aggregation_outputs
from openneuro_voxels.config import OutputConfig, ScanConfig
from openneuro_voxels.database import VoxelScanDatabase
from openneuro_voxels.plotting import write_plots
from openneuro_voxels.reporting import write_reports
from openneuro_voxels.s3 import anonymous_openneuro_client, parse_dataset_file, unique_datasets
from openneuro_voxels.scan import discover as discover_objects
from openneuro_voxels.scan import discover_and_scan, scan_database
from openneuro_voxels.validation import sample_successful_keys, validate_sample

app = typer.Typer(help="Characterize raw BOLD voxel resolution in public OpenNeuro datasets.")
console = Console()


def _dataset_args(
    dataset: list[str] | None,
    dataset_file: Path | None,
) -> list[str] | None:
    values: list[str] = []
    if dataset:
        values.extend(dataset)
    if dataset_file:
        values.extend(parse_dataset_file(str(dataset_file)))
    return unique_datasets(values) or None


@app.command()
def init_db(database: Path = typer.Option(Path("results/openneuro_voxel_scan.sqlite"))) -> None:
    """Create or migrate the SQLite audit database."""

    db = VoxelScanDatabase(database)
    db.initialize()
    db.close()
    console.print(f"Initialized {database}")


@app.command()
def discover(
    database: Path = typer.Option(Path("results/openneuro_voxel_scan.sqlite")),
    dataset: list[str] | None = typer.Option(None, help="Dataset accession to include."),
    dataset_file: Path | None = typer.Option(None, help="Text file of dataset accessions."),
    max_datasets: int | None = typer.Option(None),
    bucket: str = typer.Option("openneuro.org"),
    region: str = typer.Option("us-east-1"),
    timeout: int = typer.Option(30),
) -> None:
    """Discover raw BIDS BOLD objects without reading NIfTI headers."""

    client = anonymous_openneuro_client(region_name=region, timeout=timeout)
    stats = discover_objects(
        database_path=database,
        client=client,
        bucket=bucket,
        datasets=_dataset_args(dataset, dataset_file),
        max_datasets=max_datasets,
    )
    console.print(stats)


@app.command()
def scan(
    database: Path = typer.Option(Path("results/openneuro_voxel_scan.sqlite")),
    workers: int = typer.Option(16),
    requests_per_second: float | None = typer.Option(None),
    max_range_bytes: int = typer.Option(2 * 1024 * 1024),
    retries: int = typer.Option(3),
    timeout: int = typer.Option(30),
    dataset: list[str] | None = typer.Option(None),
    dataset_file: Path | None = typer.Option(None),
    resume: bool = typer.Option(True),
    force: bool = typer.Option(False),
    fallback: str = typer.Option("none", help="none, s3-full, or datalad"),
    bucket: str = typer.Option("openneuro.org"),
    region: str = typer.Option("us-east-1"),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Read partial NIfTI headers for discovered objects."""

    if fallback != "none":
        raise typer.BadParameter(
            "Fallback modes are explicit validation tools, not full-scan defaults."
        )
    config = ScanConfig(
        database=database,
        bucket=bucket,
        region=region,
        workers=workers,
        requests_per_second=requests_per_second,
        max_range_bytes=max_range_bytes,
        retries=retries,
        timeout=timeout,
        resume=resume,
        force=force,
        fallback=fallback,
        log_level=log_level,
    )
    stats = scan_database(config=config, datasets=_dataset_args(dataset, dataset_file))
    console.print(stats)


@app.command()
def aggregate(
    database: Path = typer.Option(Path("results/openneuro_voxel_scan.sqlite")),
    results_dir: Path = typer.Option(Path("results")),
) -> None:
    """Write file-level and dataset-resolution summary tables."""

    outputs = write_aggregation_outputs(
        database=database, config=OutputConfig(results_dir=results_dir)
    )
    console.print({name: str(path) for name, path in outputs.items()})


@app.command()
def plot(
    summary_csv: Path = typer.Option(Path("results/dataset_resolution_summary.csv")),
    figures_dir: Path = typer.Option(Path("figures")),
) -> None:
    """Generate primary publication figures from the dataset-resolution summary."""

    outputs = write_plots(summary_csv=summary_csv, figures_dir=figures_dir)
    console.print([str(path) for path in outputs])


@app.command()
def report(
    database: Path = typer.Option(Path("results/openneuro_voxel_scan.sqlite")),
    results_dir: Path = typer.Option(Path("results")),
) -> None:
    """Generate scan summary JSON and file-level error CSV."""

    outputs = write_reports(database=database, results_dir=results_dir)
    console.print({name: str(path) for name, path in outputs.items()})


@app.command()
def validate(
    key: list[str] | None = typer.Option(None, help="S3 key to validate with full-file GET."),
    key_file: Path | None = typer.Option(None, help="Text file of S3 keys to validate."),
    database: Path | None = typer.Option(
        None, help="Scan database to sample successful keys from."
    ),
    sample_size: int = typer.Option(25, help="Number of successful files to sample."),
    seed: int = typer.Option(20260616, help="Random seed for database sampling."),
    output_csv: Path = typer.Option(Path("results/validation_sample.csv")),
    bucket: str = typer.Option("openneuro.org"),
    region: str = typer.Option("us-east-1"),
    timeout: int = typer.Option(30),
    max_range_bytes: int = typer.Option(2 * 1024 * 1024),
) -> None:
    """Explicitly download selected files and compare partial vs full headers."""

    keys = list(key or [])
    if key_file:
        keys.extend(
            line.strip()
            for line in key_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
    if database:
        keys.extend(sample_successful_keys(database=database, sample_size=sample_size, seed=seed))
    if not keys:
        raise typer.BadParameter(
            "Provide --key, --key-file, or --database for explicit full-file validation."
        )
    client = anonymous_openneuro_client(region_name=region, timeout=timeout)
    frame = validate_sample(
        keys=keys,
        client=client,
        bucket=bucket,
        max_range_bytes=max_range_bytes,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    console.print(f"Wrote {output_csv}")


@app.command("all")
def run_all(
    database: Path = typer.Option(Path("results/openneuro_voxel_scan.sqlite")),
    workers: int = typer.Option(16),
    requests_per_second: float | None = typer.Option(None),
    max_range_bytes: int = typer.Option(2 * 1024 * 1024),
    retries: int = typer.Option(3),
    timeout: int = typer.Option(30),
    dataset: list[str] | None = typer.Option(None),
    dataset_file: Path | None = typer.Option(None),
    max_datasets: int | None = typer.Option(None),
    resume: bool = typer.Option(True),
    force: bool = typer.Option(False),
    fallback: str = typer.Option("none"),
    bucket: str = typer.Option("openneuro.org"),
    region: str = typer.Option("us-east-1"),
) -> None:
    """Discover, scan, aggregate, plot, and report in sequence."""

    if fallback != "none":
        raise typer.BadParameter("Full-file fallback must be requested through validate.")
    datasets = _dataset_args(dataset, dataset_file)
    config = ScanConfig(
        database=database,
        bucket=bucket,
        region=region,
        workers=workers,
        requests_per_second=requests_per_second,
        max_range_bytes=max_range_bytes,
        retries=retries,
        timeout=timeout,
        resume=resume,
        force=force,
        fallback=fallback,
    )
    stats = discover_and_scan(config=config, datasets=datasets, max_datasets=max_datasets)
    console.print(stats)
    write_aggregation_outputs(database=database)
    write_reports(database=database)
    write_plots(summary_csv=Path("results/dataset_resolution_summary.csv"))


if __name__ == "__main__":
    app()
