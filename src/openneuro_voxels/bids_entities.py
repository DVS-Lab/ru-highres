"""BIDS path filtering and lightweight entity parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

BIDS_ENTITY_ORDER = (
    "subject",
    "session",
    "task",
    "acquisition",
    "reconstruction",
    "direction",
    "run",
    "echo",
    "part",
)

_ENTITY_PREFIXES = {
    "sub": "subject",
    "ses": "session",
    "task": "task",
    "acq": "acquisition",
    "rec": "reconstruction",
    "dir": "direction",
    "run": "run",
    "echo": "echo",
    "part": "part",
}

_EXCLUDED_COMPONENTS = {
    "derivatives",
    "sourcedata",
    "code",
    "stimuli",
    "phenotype",
}


@dataclass(frozen=True)
class BidsEntities:
    """BIDS entities relevant for raw BOLD resolution grouping."""

    subject: str | None = None
    session: str | None = None
    task: str | None = None
    acquisition: str | None = None
    reconstruction: str | None = None
    direction: str | None = None
    run: str | None = None
    echo: str | None = None
    part: str | None = None

    @property
    def is_resting_state(self) -> bool:
        return self.task == "rest"

    def as_dict(self) -> dict[str, str | None]:
        return {name: getattr(self, name) for name in BIDS_ENTITY_ORDER}


def split_dataset_accession(s3_key: str) -> tuple[str, str]:
    """Return dataset accession and path below the dataset prefix."""

    accession, sep, rest = s3_key.partition("/")
    if not sep or not accession.startswith("ds"):
        raise ValueError(f"S3 key does not begin with an OpenNeuro dataset prefix: {s3_key}")
    return accession, rest


def is_raw_bold_key(s3_key: str) -> bool:
    """Return whether an S3 key points to a raw BIDS BOLD NIfTI file."""

    path = PurePosixPath(s3_key)
    parts = path.parts
    if len(parts) < 4:
        return False
    if any(part.startswith(".") for part in parts):
        return False
    if any(part in _EXCLUDED_COMPONENTS for part in parts):
        return False
    if "func" not in parts:
        return False
    if path.name.endswith(".dtseries.nii"):
        return False
    if path.name.endswith(("_sbref.nii", "_sbref.nii.gz")):
        return False
    if not path.name.endswith(("_bold.nii", "_bold.nii.gz")):
        return False

    func_index = parts.index("func")
    prior = parts[:func_index]
    if not any(part.startswith("sub-") for part in prior):
        return False
    return parts[0] not in _EXCLUDED_COMPONENTS


def parse_bids_entities(s3_key: str) -> BidsEntities:
    """Parse BIDS entities from a raw BIDS filename and parent directories."""

    filename = PurePosixPath(s3_key).name
    if filename.endswith(".nii.gz"):
        stem = filename[:-7]
    elif filename.endswith(".nii"):
        stem = filename[:-4]
    else:
        stem = filename

    values: dict[str, str] = {}
    for token in stem.split("_"):
        prefix, sep, value = token.partition("-")
        if not sep:
            continue
        field = _ENTITY_PREFIXES.get(prefix)
        if field is not None and value:
            values[field] = value

    return BidsEntities(**{name: values.get(name) for name in BIDS_ENTITY_ORDER})
