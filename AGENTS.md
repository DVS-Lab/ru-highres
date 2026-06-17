# Repository Guidelines

This repository contains two historically separate parts:

- `psychtoolbox/`: original MATLAB/Psychtoolbox task materials.
- `src/openneuro_voxels/`: Python tooling for characterizing raw BOLD voxel
  resolution in public OpenNeuro datasets.

For Python work:

- Use Python 3.11 or newer.
- Keep the package in `src/openneuro_voxels`.
- Keep raw OpenNeuro NIfTI files, DataLad clones, range-response scratch files,
  and local caches out of Git.
- Treat `results/openneuro_voxel_scan.sqlite` and file-level Parquet exports as
  generated audit artifacts; commit only when they are intentionally small.
- The primary statistical unit is dataset accession x unique anatomical-axis
  voxel-resolution triplet, not file, subject, run, echo, or task.
- Full-file retrieval must be opt-in only and must never happen silently during
  a full OpenNeuro scan.

Before committing Python changes, run:

```bash
ruff check .
mypy src
pytest
python -m build
```
