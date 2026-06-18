from pathlib import Path

import pytest

from batwind.data import samples
from batwind.data.samples import data_dir
from batwind.data.samples import data_file
from batwind.smart_ds import SmartDs

SECONDARY_SAMPLE = "3d__var_2_n00006003.plt"
MAIN_SAMPLE = "3d__var_2_n00060005.plt"


def test_data_dir_exists():
    path = data_dir()
    assert path.name == "sample_data"
    assert path.exists()


@pytest.mark.pooch
@pytest.mark.parametrize("sample_name", [SECONDARY_SAMPLE, MAIN_SAMPLE])
def test_data_file_returns_existing_tracked_fixture(sample_name):
    path = data_file(sample_name)
    assert isinstance(path, Path)
    assert path.exists()
    assert path.name == sample_name


@pytest.mark.pooch
def test_smartds_from_file_accepts_pathlike():
    sds = SmartDs.from_file(data_file(MAIN_SAMPLE))
    assert sds.raw.title


@pytest.mark.pooch
def test_smartds_from_file_merges_nearby_stellar_aux():
    sds = SmartDs.from_file(data_file(MAIN_SAMPLE))
    assert "Star_radius_m" in sds.raw.aux
    assert sds.raw.aux["DoExtendTransitionRegion"] is True
    assert float(sds.raw.aux["TeTransitionRegionSi"]) == 2.2e5
    assert float(sds.raw.aux["DeltaTeModSi"]) == 1.0e1


@pytest.mark.pooch
def test_smartds_from_file_uses_nearby_stellar_radius_as_body_radius():
    sds = SmartDs.from_file(data_file(MAIN_SAMPLE))
    assert float(sds["RBODY [m]"]) == float(sds.raw.aux["Star_radius_m"])


def test_data_file_missing_lists_available_fixtures(monkeypatch):
    def missing(name: str) -> Path:
        raise FileNotFoundError(name)

    monkeypatch.setattr(samples, "_fetch_from_g2211_archive", missing)
    with pytest.raises(FileNotFoundError) as exc:
        data_file("__definitely_missing__.plt")

    msg = str(exc.value)
    assert "Available .plt files:" in msg
    assert SECONDARY_SAMPLE in msg
    assert MAIN_SAMPLE in msg


@pytest.mark.pooch
def test_data_file_falls_back_to_archive_fetch(monkeypatch, tmp_path):
    expected = tmp_path / "3d__var_2_n00060005.plt"
    expected.write_text("", encoding="utf-8")

    def missing(root, name):
        raise FileNotFoundError(name)

    monkeypatch.setattr(samples, "_find_in_sample_data", missing)
    monkeypatch.setattr(samples, "_fetch_from_g2211_archive", lambda name: expected)

    assert data_file(MAIN_SAMPLE) == expected
