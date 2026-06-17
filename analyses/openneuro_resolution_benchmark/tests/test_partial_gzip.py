from __future__ import annotations

import pytest
from conftest import gzip_header

from openneuro_voxels.nifti_header import (
    HeaderParseError,
    InsufficientHeaderBytes,
    parse_header_from_object_bytes,
)


def test_parse_partial_gzip_header(nifti1_header_factory) -> None:
    payload = gzip_header(nifti1_header_factory())

    record = parse_header_from_object_bytes(payload[:512], is_gzipped=True)

    assert record.header_type == "NIfTI-1"
    assert record.native_zooms == pytest.approx((2.4, 2.4, 3.0))


def test_insufficient_gzip_range_requires_retry(nifti1_header_factory) -> None:
    payload = gzip_header(nifti1_header_factory())

    with pytest.raises((InsufficientHeaderBytes, HeaderParseError)):
        parse_header_from_object_bytes(payload[:12], is_gzipped=True)


def test_malformed_gzip_data_raises() -> None:
    with pytest.raises(HeaderParseError):
        parse_header_from_object_bytes(b"not-gzip", is_gzipped=True)
