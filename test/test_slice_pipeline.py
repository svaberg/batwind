from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from batread import Dataset

import batwind.pipelines.slice as slice_pipeline
from batwind.smart_ds import SmartDs
from batwind.visualisation.slice import plot_xz_slice_tripcolor_with_cross_quantiles


def test_process_plt_file_passes_tripcolor_kwargs(monkeypatch, tmp_path):
    file_path = tmp_path / "x=0_demo.plt"
    file_path.write_text("", encoding="utf-8")

    calls = []

    monkeypatch.setattr(slice_pipeline.SmartDs, "from_file", staticmethod(lambda *args, **kwargs: object()))

    def fake_plot(ds, *, var, tripcolor_kwargs=None, **kwargs):
        calls.append(
            {
                "ds": ds,
                "var": var,
                "tripcolor_kwargs": tripcolor_kwargs,
                "extra_kwargs": kwargs,
            }
        )
        fig, ax = plt.subplots()
        ax.plot([0.0, 1.0], [0.0, 1.0])
        return fig, (ax,), None

    monkeypatch.setattr(slice_pipeline, "plot_xz_slice_tripcolor_with_cross_quantiles", fake_plot)

    slice_pipeline.process_plt_file(file_path)

    assert [call["var"] for call in calls] == [
        "Rho [kg/m^3]",
        "U [m/s]",
        "B [T]",
        "B_r [T]",
    ]
    assert all(call["extra_kwargs"] == {} for call in calls)
    assert calls[0]["tripcolor_kwargs"]["norm"].__class__.__name__ == "LogNorm"
    assert calls[1]["tripcolor_kwargs"] == {"shading": "flat"}
    assert calls[2]["tripcolor_kwargs"]["norm"].__class__.__name__ == "LogNorm"
    assert calls[3]["tripcolor_kwargs"]["cmap"] == "RdBu_r"
    assert calls[3]["tripcolor_kwargs"]["norm"].__class__.__name__ == "SymLogNorm"
    assert (tmp_path / "slice" / "x_0_demo.slices.rho.png").exists()
    assert (tmp_path / "slice" / "x_0_demo.slices.u.png").exists()
    assert (tmp_path / "slice" / "x_0_demo.slices.b.png").exists()
    assert (tmp_path / "slice" / "x_0_demo.slices.br.png").exists()


def test_plot_xz_slice_tripcolor_with_cross_quantiles_accepts_smartds(tmp_path):
    variables = ["X [R]", "Y [R]", "Z [R]", "Q [none]"]
    points = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [1.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 1.0, 4.0],
        ],
        dtype=float,
    )
    corners = np.array([[0, 1, 2, 3]], dtype=int)
    smart_ds = SmartDs(Dataset(points, corners, aux={}, title="slice", variables=variables, zone="zslice"))

    fig, _axes, cbar = plot_xz_slice_tripcolor_with_cross_quantiles(
        smart_ds,
        var="Q [none]",
        tripcolor_kwargs={"shading": "flat"},
    )
    try:
        assert cbar.ax.get_ylabel() == "Q [none]"
        out_path = tmp_path / "smartds_slice.png"
        fig.savefig(out_path)
        assert out_path.exists()
    finally:
        plt.close(fig)
