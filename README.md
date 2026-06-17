# Rutgers High-Resolution Reward fMRI

This repository supports the manuscript:

**Differential representations for affective and informative components of
reward in the striatum and hippocampus**

The study uses high-resolution task fMRI and representational similarity
analysis to examine affective and informative reward representations in
subcortical regions, especially dorsal/ventral striatum and anterior/posterior
hippocampus.

The manuscript reports functional BOLD acquisition at 3T with single-shot
T2*-weighted EPI, GRAPPA R=2, TR=2000 ms, TE=28 ms, matrix 128 x 128, FOV 204
mm, and 1.75 x 1.75 mm in-plane voxels. The high-resolution acquisition is
important for the paper's central goal: resolving representational structure
within relatively small subcortical regions.

## Repository Layout

- `psychtoolbox/`: MATLAB/Psychtoolbox task code.
- `docs/prereg/`: preregistration figures.
- `analyses/openneuro_resolution_benchmark/`: reproducible OpenNeuro analysis
  used to contextualize the study's BOLD spatial resolution for reviewer
  response and future figures.

## OpenNeuro Resolution Benchmark

The OpenNeuro benchmark is intentionally organized as a supporting analysis,
not as the identity of this repository. It scans raw public BOLD NIfTI headers
from OpenNeuro without full-file downloads under normal operation, aggregates
to one observation per dataset x unique spatial-resolution triplet, and
generates figures/tables for comparing this study's acquisition with public
datasets.

See:

```text
analyses/openneuro_resolution_benchmark/README.md
```

The next analysis target is to stratify OpenNeuro BOLD resolution by acquisition
type, especially likely single-band/conventional EPI versus multiband/SMS
datasets. That distinction matters because multiband acceleration can support
different spatial/temporal tradeoffs, while the reviewer concern is most
directly addressed by comparing against regular single-band task fMRI studies.

## Collaboration Note

The benchmark and reviewer-response figures are being developed here first.
Once the figures and text are settled, the relevant materials can be merged
into Karen Shen's repository:

```text
https://github.com/KarenShen21/ru-highres
```
