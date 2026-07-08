"""Unit tests for the post-processor — run with: pytest tests/ -q"""

from core.parser import (build_record, clean_text, compute_flag,
                         normalize_value, parse_range)


def test_clean_text_strips_noise():
    assert clean_text("  Hemoglobin :") == "Hemoglobin"
    assert clean_text("Page 2 of 3") == ""
    assert clean_text("www.somelab.com") == ""


def test_normalize_value_fixes_ocr_digits():
    assert normalize_value("l3.4") == "13.4"
    assert normalize_value("1O5") == "105"
    assert normalize_value("Positive") == "Positive"  # non-numeric untouched


def test_parse_range_and_flag():
    assert parse_range("13.0 - 17.0") == (13.0, 17.0)
    assert compute_flag("12.1", "13.0-17.0") == "LOW"
    assert compute_flag("18.2", "13.0-17.0") == "HIGH"
    assert compute_flag("14.5", "13.0-17.0") == "NORMAL"
    assert compute_flag("", "13-17") == ""


def test_build_record_assembles_rows():
    dets = [
        {"class": "patient_name", "text": "Ravi Kumar", "box": [10, 10, 200, 40]},
        {"class": "test_name", "text": "Hemoglobin", "box": [10, 100, 200, 130]},
        {"class": "value", "text": "13.4", "box": [220, 100, 280, 130]},
        {"class": "unit", "text": "g/dL", "box": [300, 100, 360, 130]},
        {"class": "reference_range", "text": "13.0-17.0",
         "box": [380, 100, 480, 130]},
        {"class": "test_name", "text": "Platelets", "box": [10, 160, 200, 190]},
        {"class": "value", "text": "250000", "box": [220, 160, 300, 190]},
    ]
    record = build_record(dets)
    assert record["header"]["patient_name"] == "Ravi Kumar"
    assert len(record["rows"]) == 2
    hb = record["rows"][0]
    assert hb["value"] == "13.4" and hb["flag"] == "NORMAL"
