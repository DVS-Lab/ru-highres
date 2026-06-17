# OpenNeuro BOLD Voxel Resolution Scanner

This repository contains a reproducible Python workflow for characterizing the
spatial voxel resolution of raw BOLD fMRI NIfTI files in public OpenNeuro
datasets. The original Rutgers high-resolution task materials remain in
`psychtoolbox/`; the OpenNeuro scanner lives in `src/openneuro_voxels/`.

The central analysis unit is not a file, subject, run, echo, or task. The
primary unit is:

```text
OpenNeuro dataset accession x unique anatomical-axis voxel-resolution triplet
```

This avoids grossly overweighting large datasets. A dataset with 600 raw BOLD
files at 2.0 x 2.0 x 2.0 mm contributes one histogram observation. A dataset
with two genuinely different BOLD resolutions contributes two observations.
Repeated subjects, sessions, runs, echoes, parts, reconstructions, and tasks
collapse when the anatomical-axis resolution triplet is the same.

## What Is Scanned

The scanner reads public anonymous OpenNeuro S3 objects from:

```text
bucket: openneuro.org
region: us-east-1
```

Included files are raw BIDS functional NIfTI images matching:

```text
*_bold.nii
*_bold.nii.gz
```

The path must contain a raw BIDS `func/` directory. The scanner excludes
`derivatives/`, `sourcedata/`, `code/`, `stimuli/`, `phenotype/`, hidden Git or
DataLad paths, CIFTI derivatives, SBRefs, and non-BOLD suffixes.

## Header Strategy

Normal scans do not issue full-object GET requests for NIfTI images. For
`.nii.gz` objects, the scanner requests bytes from the beginning of the object,
decompresses only enough of the gzip stream to parse the NIfTI header, and
retries progressively larger byte ranges up to the configured maximum. For
uncompressed `.nii` files, it requests only enough bytes for a NIfTI-1 or
NIfTI-2 header.

NiBabel parses both NIfTI-1 and NIfTI-2 headers, including little- and
big-endian headers. Native spatial zooms come from `header.get_zooms()[:3]`.
The best available affine is used with NiBabel orientation utilities to map
native dimensions to RAS anatomical axes:

- `voxel_size_lr_mm`
- `voxel_size_ap_mm`
- `voxel_size_is_mm`

If the affine cannot be mapped reliably, native-axis measurements are retained
for audit/supplementary summaries, but those files are excluded from the
primary anatomical-axis histograms.

## Install

Use Python 3.11 or newer:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Commands

Discover raw BOLD objects without reading headers:

```bash
openneuro-voxels discover \
  --database results/openneuro_voxel_scan.sqlite \
  --dataset ds000001
```

Scan discovered objects with resumable state:

```bash
openneuro-voxels scan \
  --database results/openneuro_voxel_scan.sqlite \
  --workers 16 \
  --resume
```

Run the normal pipeline:

```bash
openneuro-voxels all \
  --database results/openneuro_voxel_scan.sqlite \
  --workers 16 \
  --resume
```

Regenerate outputs from an existing SQLite audit database:

```bash
openneuro-voxels aggregate --database results/openneuro_voxel_scan.sqlite
openneuro-voxels report --database results/openneuro_voxel_scan.sqlite
openneuro-voxels plot --summary-csv results/dataset_resolution_summary.csv
```

Explicit full-file validation is opt-in:

```bash
openneuro-voxels validate \
  --database results/openneuro_voxel_scan.sqlite \
  --sample-size 25 \
  --seed 20260616 \
  --output-csv results/validation_sample.csv
```

## Outputs

Generated outputs are written to `results/` and `figures/`.

Primary tabular outputs:

- `results/dataset_resolution_summary.csv`
- `results/dataset_resolution_summary.parquet`
- `results/dataset_resolution_sensitivity.csv`
- `results/common_resolution_triplets.csv`
- `results/scan_summary.json`
- `results/file_level_errors.csv`

Large audit outputs are intentionally ignored by Git for full scans:

- `results/openneuro_voxel_scan.sqlite`
- `results/file_level_headers.parquet`

Primary plots:

- `figures/hist_voxel_size_lr.{png,pdf}`
- `figures/hist_voxel_size_ap.{png,pdf}`
- `figures/hist_voxel_size_is.{png,pdf}`
- `figures/hist_voxel_size_all_dimensions.{png,pdf}`
- `figures/common_resolution_triplets.{png,pdf}`
- `figures/resolutions_per_dataset.{png,pdf}`

The primary histograms use `dataset_resolution_summary.csv`. Each row has
weight one. They are not weighted by number of files, subjects, runs, echoes, or
tasks.

## Current Pilot Provenance

The committed pilot outputs were generated from the public OpenNeuro S3 object
tree on 2026-06-17 UTC. The SQLite audit rows record header extraction with
package version `0.1.0` at Git commit
`b107b8ceef2ecb41cef7dc0059f995b51a799f10`; later commits on this branch add
documentation and reporting/validation refinements.

Pilot datasets examined:

- `ds000001`
- `ds000005`
- `ds000102`
- `ds000117`
- `ds000210`
- `ds000246` (no matching raw BOLD files found)

Pilot summary:

- Dataset prefixes examined: 6
- Datasets containing raw BOLD data: 5
- Raw BOLD files discovered: 756
- Headers successfully parsed: 756
- Failed headers: 0
- Failure percentage: 0.0%
- Compressed bytes retrieved for header extraction: 6,193,152
- Dataset-resolution observations: 8
- Datasets with one resolution: 3
- Datasets with multiple resolutions: 2
- QC flags: none

The pilot includes task datasets, a resting-state dataset, a multi-task dataset,
and a multi-echo dataset. `ds000210` contains both `rest` and `cuedSGT` task
labels and echo labels `1`, `2`, and `3`.

Resume behavior was checked by rerunning `openneuro-voxels scan --resume`
against the pilot database. The rerun reported:

```text
requested=0, successes=0, failures=0
```

This demonstrates that already-successful key/ETag pairs were skipped instead
of duplicated.

## Validation

The validation command sampled 25 successfully parsed pilot files using seed
`20260616`, explicitly downloaded those full NIfTI files to a temporary
directory, and compared partial-header native zooms with `nibabel.load()` on the
complete local files.

Validation result:

```text
25/25 passed within 1e-6 mm
maximum absolute difference: 0.0 mm
```

The validation details are in `results/validation_sample.csv`.

## Aggregation Rules

The primary grouping triplet is the canonical rounded anatomical-axis triplet:

```text
voxel_size_lr_mm x voxel_size_ap_mm x voxel_size_is_mm
```

Values are rounded to 0.001 mm for grouping. Unrounded minima and maxima are
reported for each group. Multiple tasks with the same resolution collapse to one
row. Multi-echo data at one resolution collapse to one row. Different
resolution triplets in the same dataset remain separate, even when voxel volume
is equal.

The sensitivity summary excludes groups represented by only one file or by less
than 5% of a dataset's BOLD files. It is supplementary and does not replace the
complete primary summary.

## Inspecting Failures

Failed header reads are recorded in SQLite and exported to:

```text
results/file_level_errors.csv
```

Failures are stored with dataset accession, key, error class, message,
compressed bytes retrieved, retry count, and timestamp. A malformed or
unavailable object does not stop the full scan.

## Methods

We enumerated public OpenNeuro S3 prefixes matching `ds[0-9A-Za-z]+/` and
recursively listed objects within selected dataset prefixes. Objects were
retained when their paths matched raw BIDS functional BOLD NIfTI files and did
not occur under excluded derivative, source, code, stimulus, phenotype, hidden,
or non-functional paths. For each retained object, we requested a byte range
beginning at byte zero. Gzip-compressed NIfTI files were decompressed only until
the NIfTI header could be parsed; uncompressed NIfTI files were parsed directly
from the header bytes. NiBabel header classes extracted NIfTI version, image
shape, zooms, qform/sform codes, affine information, and orientation. Native
voxel dimensions were mapped to RAS anatomical axes using the best available
affine. File-level records were stored in SQLite with key, ETag, LastModified,
BIDS entities, retrieval metadata, header fields, QC flags, and errors.
Dataset-level summaries collapsed files to one row per dataset accession and
unique rounded anatomical-axis spatial-resolution triplet. Histograms used those
dataset-resolution rows as equally weighted observations.

## Limitations

OpenNeuro S3 represents the public object tree at scan time; datasets can change
afterward. Ambiguous, missing, singular, or internally inconsistent affines are
flagged and excluded from primary anatomical-axis histograms unless a documented
fallback rule is applied. Malformed headers, unavailable objects, throttling, or
transient network failures are recorded as file-level errors rather than
terminating the scan. Full-file downloads are intentionally disabled during
normal scans and are available only through the explicit validation command.
