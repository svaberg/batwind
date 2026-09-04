from pathlib import Path

import matplotlib.pyplot as plt

from batwind.pipelines.utils import annotate_iteration_axis
from batwind.pipelines.utils import iteration_token_from_path


def test_iteration_token_from_path_extracts_batsrus_iteration():
    assert iteration_token_from_path("shl_var_1_n00004740.plt") == "n00004740"
    assert iteration_token_from_path(Path("slice/x_0_var_2_n00108000.slices.rho.png")) == "n00108000"


def test_iteration_token_from_path_returns_none_without_iteration():
    assert iteration_token_from_path("3DALL_t000000_000000.bin") is None


def test_annotate_iteration_axis_adds_iteration_text_inside_axes():
    fig, ax = plt.subplots()
    try:
        annotate_iteration_axis(ax, "shl_var_1_n00004740.plt")
        assert [text.get_text() for text in ax.texts] == ["n00004740"]
    finally:
        plt.close(fig)
