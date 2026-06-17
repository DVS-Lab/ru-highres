from __future__ import annotations

import gzip

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def nifti1_header_factory():
    def factory(
        *,
        shape: tuple[int, ...] = (4, 5, 6, 7),
        zooms: tuple[float, ...] = (2.4, 2.4, 3.0, 1.2),
        affine: np.ndarray | None = None,
        endian: str = "<",
    ) -> bytes:
        header = nib.Nifti1Header()
        header.set_data_shape(shape)
        header.set_zooms(zooms)
        if affine is None:
            affine = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
        header.set_sform(affine, code=1)
        try:
            header.set_qform(affine, code=1)
        except Exception:
            header.set_qform(None, code=0)
        if endian == ">":
            header = header.as_byteswapped()
        return header.binaryblock

    return factory


@pytest.fixture
def nifti2_header_factory():
    def factory(
        *,
        shape: tuple[int, ...] = (3, 4, 5),
        zooms: tuple[float, ...] = (1.5, 2.0, 2.5),
        affine: np.ndarray | None = None,
    ) -> bytes:
        header = nib.Nifti2Header()
        header.set_data_shape(shape)
        header.set_zooms(zooms)
        if affine is None:
            affine = np.diag([zooms[0], zooms[1], zooms[2], 1.0])
        header.set_sform(affine, code=1)
        header.set_qform(affine, code=1)
        return header.binaryblock

    return factory


def gzip_header(header_bytes: bytes) -> bytes:
    return gzip.compress(header_bytes + b"\0" * 1024)
