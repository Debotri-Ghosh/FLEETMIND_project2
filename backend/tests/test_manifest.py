from fleetmind.ingestion.manifest import (
    guess_diameter,
    guess_fault_type,
    guess_load,
    is_target_file,
)


def test_is_target_file_keeps_12k_de():
    p = ["data", "raw", "cwru", "CWRU_organized", "12k_DE", "Load_0", "Ball", "B007_0.mat"]
    assert is_target_file(type("P", (), {"parts": tuple(p)})()) is True


def test_is_target_file_excludes_48k():
    p = ["data", "raw", "cwru", "CWRU_organized", "48k_DE", "Load_0", "Ball", "B007_0.mat"]
    assert is_target_file(type("P", (), {"parts": tuple(p)})()) is False


def test_guess_fault_type():
    p = ["12k_DE", "Load_0", "InnerRace", "IR007_0.mat"]
    assert guess_fault_type(type("P", (), {"parts": tuple(p)})()) == "InnerRace"


def test_guess_diameter():
    assert guess_diameter("IR007_0.mat", "InnerRace") == "0.007"
    assert guess_diameter("Normal_0.mat", "Normal") == ""
    assert guess_diameter("bad.mat", "Ball") == "NEEDS_REVIEW"


def test_guess_load():
    p = ["12k_DE", "Load_2", "Ball", "B007_2.mat"]
    assert guess_load(type("P", (), {"parts": tuple(p)})()) == "2"
