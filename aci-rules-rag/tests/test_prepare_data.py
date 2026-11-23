import csv
import pytest
from src.common import ProtocolError
from src.prepare_data import load_csv, normalize, validate_records


def test_loading_and_normalization(tmp_path):
    path = tmp_path / "sample.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "encounter_id", "dialogue", "note"])
        writer.writerow(["synthetic", "one", "[patient] cafe\u0301\r\nNo pain.", "A reference."])
    rows = load_csv(path, "train")
    assert rows[0]["transcript"] == "[patient] café\nNo pain."
    assert validate_records(rows, {"train": 1})["counts"] == {"train": 1}


def test_duplicates_are_detected(records):
    rows = [{**r, "reference_note": str(i)} for i, r in enumerate(records)]
    report = validate_records(rows, {"test1": 3})
    assert len(report["duplicate_texts"]) == 2
    rows[1]["encounter_id"] = rows[0]["encounter_id"]
    with pytest.raises(ProtocolError, match="Duplicate encounter"):
        validate_records(rows, {"test1": 3})


def test_corrupt_and_empty_csv(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("dataset,encounter_id,dialogue,note\nx,id,,note\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="Empty"):
        load_csv(path, "train")


def test_wrong_split_counts(records):
    rows = [{**r, "reference_note": str(i)} for i, r in enumerate(records)]
    with pytest.raises(ProtocolError, match="counts"):
        validate_records(rows, {"test1": 40})


def test_no_clinical_rewriting():
    assert normalize("  A  B\rC ") == "  A  B\nC "
