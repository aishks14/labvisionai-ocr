"""Exporter smoke tests — every format must produce non-empty bytes."""

import json

import pytest

from core.exporters import export

RECORD = {
    "header": {"patient_name": "Ravi Kumar", "age": "34", "gender": "M"},
    "rows": [{"test_name": "Hemoglobin", "value": "13.4", "unit": "g/dL",
              "reference_range": "13.0-17.0", "flag": "NORMAL"}],
}


@pytest.mark.parametrize("fmt", ["csv", "json", "xlsx", "pdf", "fhir"])
def test_every_format_produces_bytes(fmt):
    data, mime, suffix = export(RECORD, fmt)
    assert isinstance(data, bytes) and len(data) > 20
    assert mime and suffix


def test_fhir_is_valid_bundle():
    data, _, _ = export(RECORD, "fhir")
    bundle = json.loads(data)
    assert bundle["resourceType"] == "Bundle"
    kinds = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert "DiagnosticReport" in kinds and "Observation" in kinds


def test_unknown_format_rejected():
    with pytest.raises(ValueError):
        export(RECORD, "docx")
