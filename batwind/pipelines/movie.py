"""Movie generation for pipeline PNG frame series."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

log = logging.getLogger(__name__)

DEFAULT_MOVIE_FPS = 12
_INDEXED_FRAME_PATTERN = re.compile(r"_n\d+")
_LONG_DIGITS_PATTERN = re.compile(r"\d{4,}")


@dataclass(frozen=True)
class PngMovieSeries:
    """One ordered PNG frame series and its target movie path."""

    movie_path: Path
    frame_paths: tuple[Path, ...]


def _collapse_movie_stem(text: str) -> str:
    stem = str(text)
    while "__" in stem:
        stem = stem.replace("__", "_")
    while ".." in stem:
        stem = stem.replace("..", ".")
    while "_." in stem:
        stem = stem.replace("_.", ".")
    while "._" in stem:
        stem = stem.replace("._", ".")
    return stem.strip("._-") or "movie"


def _frame_token_span(frame_stem: str) -> tuple[int, int] | None:
    indexed = _INDEXED_FRAME_PATTERN.search(frame_stem)
    if indexed is not None:
        return indexed.span()
    long_digit_matches = list(_LONG_DIGITS_PATTERN.finditer(frame_stem))
    if long_digit_matches:
        return long_digit_matches[-1].span()
    return None


def infer_movie_path_for_frame(frame_path: str | Path) -> Path | None:
    """Infer the aggregate movie path for one time-indexed PNG frame."""
    path = Path(frame_path)
    span = _frame_token_span(path.stem)
    if span is None:
        return None
    start, end = span
    movie_stem = _collapse_movie_stem(path.stem[:start] + path.stem[end:])
    return path.with_name(f"{movie_stem}.mp4")


def collect_recorded_png_movie_series(
    directory: str | Path,
    computed_results: dict[str, dict[str, object]],
) -> list[PngMovieSeries]:
    """Collect ordered PNG frame series from recorder payloads."""
    root = Path(directory)
    grouped: dict[Path, list[Path]] = {}

    for file_key, payload in computed_results.items():
        if not isinstance(payload, dict):
            continue
        file_parent = Path(str(file_key)).parent
        for field_name, field_payload in payload.items():
            if field_name == "meta":
                continue
            value = field_payload.get("value") if isinstance(field_payload, dict) else field_payload
            if not isinstance(value, str) or not value.lower().endswith(".png"):
                continue
            frame_path = (root / file_parent / value).resolve()
            movie_path = infer_movie_path_for_frame(frame_path)
            if movie_path is None:
                continue
            grouped.setdefault(movie_path, []).append(frame_path)

    series: list[PngMovieSeries] = []
    for movie_path, frame_paths in grouped.items():
        unique_existing = sorted({path for path in frame_paths if path.exists()}, key=lambda path: str(path))
        if len(unique_existing) < 2:
            continue
        series.append(PngMovieSeries(movie_path=movie_path, frame_paths=tuple(unique_existing)))

    series.sort(key=lambda item: str(item.movie_path))
    return series


def write_png_movie_series(
    movie_path: str | Path,
    frame_paths: tuple[Path, ...] | list[Path],
    *,
    fps: int = DEFAULT_MOVIE_FPS,
) -> Path | None:
    """Render one ordered PNG frame series to an MP4 movie with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        log.warning("Skipping movie generation because ffmpeg is not available: %s", movie_path)
        return None

    output_path = Path(movie_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="batwind-movie-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        for frame_id, frame_path in enumerate(frame_paths):
            link_path = tmp_dir / f"frame_{frame_id:06d}.png"
            os.symlink(Path(frame_path).resolve(), link_path)

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(int(fps)),
                "-i",
                str(tmp_dir / "frame_%06d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=True,
        )

    log.info("Wrote movie %s from %d PNG frames.", output_path.name, len(frame_paths))
    return output_path


def write_recorded_png_movies(
    directory: str | Path,
    computed_results: dict[str, dict[str, object]],
    *,
    fps: int = DEFAULT_MOVIE_FPS,
) -> list[Path]:
    """Render MP4 movies for all recorded multi-frame PNG series."""
    output_paths: list[Path] = []
    for series in collect_recorded_png_movie_series(directory, computed_results):
        written = write_png_movie_series(series.movie_path, series.frame_paths, fps=int(fps))
        if written is not None:
            output_paths.append(written)
    return output_paths
