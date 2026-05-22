from pathlib import Path

import logging
import pytest

from batwind.param_in import ParamIn
from batwind.param_in import flatten_includes
from batwind.param_in import find_param_in
from batwind.param_in import stellar_aux_from_nearby_param_in


SAMPLE_PARAM_IN = Path("sample_data/PARAM.in")
MAIN_SAMPLE = "sample_data/3d__var_2_n00060005.plt"


def test_flatten_includes_expands_existing_children(tmp_path):
    child = tmp_path / "child.in"
    child.write_text("#STAR\n1.0 RadiusStar\n", encoding="utf-8")

    parent = tmp_path / "PARAM.in"
    parent.write_text("#INCLUDE\nchild.in\n#GRID\n1 nRootBlock1\n", encoding="utf-8")

    flat = flatten_includes(parent)

    assert "#INCLUDE" not in flat
    assert "child.in" not in flat
    assert "#STAR" in flat
    assert "#GRID" in flat


def test_param_in_preserves_components_sessions_and_duplicate_commands(tmp_path):
    config_file = tmp_path / "PARAM.in"
    config_file.write_text(
        "\n".join(
            [
                "Begin session: 1",
                "#AMRREGION",
                "Inner NameRegion",
                "#AMRREGION",
                "Outer NameRegion",
                "#BEGIN_COMP SC",
                "#COORDSYSTEM",
                "HGR TypeCoordSystem",
                "#END_COMP",
                "#RUN",
                "#GRID",
                "1 nRootBlock1",
            ]
        ),
        encoding="utf-8",
    )

    config = ParamIn.from_file(config_file)

    assert config.num_sessions() == 2
    assert config.get_param("#COORDSYSTEM", 0, component="SC", session=0) == "HGR"
    assert len(config.get_commands("#AMRREGION", session=0)) == 2
    assert config.get_param("#AMRREGION", 0, session=0, occurrence=0) == "Inner"
    assert config.get_param("#AMRREGION", 0, session=0, occurrence=1) == "Outer"
    assert config.get_param("#GRID", 0, session=1) == 1


def test_split_value_and_label_only_splits_on_tab_or_three_spaces(tmp_path):
    config_file = tmp_path / "PARAM.in"
    config_file.write_text(
        "\n".join(
            [
                "#TEST",
                "1 2  3",
                "4   FourLabel",
                "5\tFiveLabel",
            ]
        ),
        encoding="utf-8",
    )

    config = ParamIn.from_file(config_file)
    params = config.get_named_params("#TEST")

    assert params["param_0"] == "1 2  3"
    assert params["FourLabel"] == 4
    assert params["FiveLabel"] == 5


@pytest.mark.pooch
def test_param_in_parses_sample_file():
    config = ParamIn.from_file(SAMPLE_PARAM_IN)

    assert config.num_sessions() >= 1
    assert config.get_param("#COMPONENT", 0) == "SC"
    assert config.get_named_params("#STAR")["NameStar"] == "tau Boötis"
    assert config.get_named_params("#POYNTINGFLUX")["PoyntingFluxPerBSi"] == 1e6
    assert config.get_named_params("#STARTTIME")["iYear"] == 2011
    assert config.get_named_params("#COORDSYSTEM")["TypeCoordSystem"] == "HGR"
    assert config.get_named_params("#GRID")["xMin"] == -200.0
    assert config.get_named_params("#GRID")["zMax"] == 200.0
    assert len(config.get_commands("#AMRREGION")) == 2
    assert config.get_named_params("#AMRREGION", occurrence=0)["NameRegion"] == "InnerShell"
    assert config.get_named_params("#AMRREGION", occurrence=1)["NameRegion"] == "LargeShell"


@pytest.mark.pooch
def test_param_in_extracts_stellar_params_and_nearby_lookup():
    config = ParamIn.from_file(SAMPLE_PARAM_IN)
    star = config.stellar_params()
    nearby = stellar_aux_from_nearby_param_in(MAIN_SAMPLE)

    assert star["Star_name"] == "tau Boötis"
    assert star["Star_radius_m"] > 1.0e9
    assert star["Star_mass_kg"] > 1.0e30
    assert star["Star_rotational_period_s"] > 0.0
    assert star["Star_rotation_rate_rad_s"] > 0.0
    assert nearby["Star_name"] == star["Star_name"]
    assert nearby["Star_radius_m"] == star["Star_radius_m"]


def test_find_param_in_checks_parent_chain_and_logs_choice(tmp_path, caplog):
    data_dir = tmp_path / "a" / "b" / "c"
    data_dir.mkdir(parents=True)
    data_file = data_dir / "data.plt"
    data_file.write_text("", encoding="utf-8")
    param_file = tmp_path / "a" / "PARAM.in"
    param_file.write_text("#STAR\n1.0 RadiusStar\n1.0 MassStar\n25.0 RotationPeriodStar\n", encoding="utf-8")

    with caplog.at_level(logging.INFO):
        found = find_param_in(data_file)

    assert found == param_file
    assert f"Using PARAM.in {param_file}" in caplog.text
