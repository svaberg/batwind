from pathlib import Path

import batwind.pipelines.movie as movie


def test_collect_recorded_png_movie_series_groups_numbered_frames(tmp_path):
    frame_a = tmp_path / "slice" / "x_0_var_2_n00000000.slices.rho.png"
    frame_b = tmp_path / "slice" / "x_0_var_2_n00000010.slices.rho.png"
    frame_c = tmp_path / "slice" / "z_0_var_3_n00000000.slices.rho.png"
    for path in (frame_a, frame_b, frame_c):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    computed_results = {
        "x=0_var_2_n00000000.plt": {
            "slice_rho_png": {"value": "slice/x_0_var_2_n00000000.slices.rho.png"},
        },
        "x=0_var_2_n00000010.plt": {
            "slice_rho_png": {"value": "slice/x_0_var_2_n00000010.slices.rho.png"},
        },
        "z=0_var_3_n00000000.plt": {
            "slice_rho_png": {"value": "slice/z_0_var_3_n00000000.slices.rho.png"},
        },
    }

    grouped = movie.collect_recorded_png_movie_series(tmp_path, computed_results)

    assert len(grouped) == 1
    assert grouped[0].movie_path == (tmp_path / "slice" / "x_0_var_2.slices.rho.mp4").resolve()
    assert grouped[0].frame_paths == (frame_a.resolve(), frame_b.resolve())


def test_write_recorded_png_movies_invokes_ffmpeg(monkeypatch, tmp_path):
    frame_a = tmp_path / "slice" / "x_0_var_2_n00000000.slices.rho.png"
    frame_b = tmp_path / "slice" / "x_0_var_2_n00000010.slices.rho.png"
    for path in (frame_a, frame_b):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    computed_results = {
        "x=0_var_2_n00000000.plt": {
            "slice_rho_png": {"value": "slice/x_0_var_2_n00000000.slices.rho.png"},
        },
        "x=0_var_2_n00000010.plt": {
            "slice_rho_png": {"value": "slice/x_0_var_2_n00000010.slices.rho.png"},
        },
    }

    commands = []

    monkeypatch.setattr(movie.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None)

    def fake_run(command, check):
        commands.append(command)
        Path(command[-1]).write_text("", encoding="utf-8")

    monkeypatch.setattr(movie.subprocess, "run", fake_run)

    written = movie.write_recorded_png_movies(tmp_path, computed_results, fps=7)

    assert written == [tmp_path / "slice" / "x_0_var_2.slices.rho.mp4"]
    assert written[0].exists()
    assert commands
    assert "-framerate" in commands[0]
    assert "7" in commands[0]
