"""Per-file log-diagnostics pipeline for ``batwind-pipe``."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)
add_record = logging.getLogger(f"recorder.{__name__}").debug

LOG_FILE_PATTERN = re.compile(r"log_n(\d+)\.log$")
METRIC_COLUMN_PATTERN = re.compile(r"(?P<metric>.+)_R=(?P<radius>[^ ]+)$")
FLUX_METRIC_PATTERN = re.compile(r"(?P<base>.+?)(?P<variant>in|out)?flx$")
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]*$")
VARIANT_ORDER = {"total": 0, "in": 1, "out": 2, "value": 3}
BOOL_TOKEN_TO_VALUE = {"T": True, "F": False}
SECTION_BOOLEAN_NAMES = {"#DOAMR": "DoAmr"}
REPORT_FLAG_NAMES = (
    "DoAmr",
    "UseHeatFluxRegion",
    "UseHeatFluxCollisionless",
    "UseZdiBoundary",
    "UseZdiBoundaryRadial",
    "UseZdiMagnetogram",
)
TRACKED_SCALAR_NAMES = {
    "MaxIteration",
    "NameZdiCoeffFile",
    "StringLogRadius",
    "StringLogVar",
    "TypeZdiBoundary",
    "TypeZdiRamp",
    "ZdiRampIterStart",
    "ZdiRampIterStop",
}
FLUX_DISPLAY_NAMES = {
    "rho": "Mass Flux",
    "jx": "Angular Momentum Flux (x)",
    "jy": "Angular Momentum Flux (y)",
    "jz": "Angular Momentum Flux",
}


@dataclass(frozen=True)
class SessionInfo:
    label: str
    start: int
    stop: int
    local_bools: dict[str, bool]
    local_scalars: dict[str, str]
    active_bools: dict[str, bool]
    active_scalars: dict[str, str]
    summary: str


def resolve_session_iteration(value: str, *, session_start: int) -> int:
    """Resolve one session-local or absolute iteration scalar to absolute iteration."""
    iteration = int(float(value))
    if session_start > 0 and iteration < session_start:
        return session_start + iteration
    return iteration


def find_run_root(directory: Path) -> Path | None:
    """Find the nearest ancestor directory containing ``PARAM.in``."""
    for candidate in (directory, *directory.parents):
        if (candidate / "PARAM.in").exists():
            return candidate
    return None


def load_log_file(path: Path) -> tuple[str, list[str], np.ndarray]:
    """Load one BATSRUS log file title, columns, and numeric data."""
    with path.open() as handle:
        title = handle.readline().strip()
        columns = handle.readline().split()
    data = np.loadtxt(path, skiprows=2)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != len(columns):
        raise ValueError(f"{path.name}: expected {len(columns)} columns, got {data.shape[1]}")
    return title, columns, data


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    """Compute modest padded y-limits for one plotted series."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    ymin = float(np.min(finite))
    ymax = float(np.max(finite))
    if ymin == ymax:
        pad = 1.0 if ymin == 0.0 else 0.05 * abs(ymin)
        return ymin - pad, ymax + pad
    pad = 0.05 * (ymax - ymin)
    return ymin - pad, ymax + pad


def positive_log_limits(values: np.ndarray) -> tuple[float, float]:
    """Compute multiplicatively padded y-limits for positive-only log panels."""
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        raise ValueError("positive_log_limits requires at least one positive value")
    ymin = float(np.min(positive))
    ymax = float(np.max(positive))
    if ymin == ymax:
        return ymin / 1.5, ymax * 1.5
    pad_factor = (ymax / ymin) ** 0.05
    return ymin / pad_factor, ymax * pad_factor


def nonzero_log_magnitudes(values: np.ndarray) -> np.ndarray:
    """Return positive magnitudes for all nonzero finite values in one series."""
    finite = values[np.isfinite(values)]
    return np.abs(finite[finite != 0.0])


def log_axis_limits(values: np.ndarray) -> tuple[float, float]:
    """Compute y-limits for one signed series drawn on a log-magnitude axis."""
    magnitudes = nonzero_log_magnitudes(values)
    if magnitudes.size == 0:
        return 1.0e-12, 1.0
    return positive_log_limits(magnitudes)


def plot_signed_log_series(
    axis: plt.Axes,
    iteration: np.ndarray,
    values: np.ndarray,
    *,
    linewidth: float,
    label: str | None = None,
) -> None:
    """Plot positive values solid and absolute negative values dashed on one log axis."""
    positive = np.where(values > 0.0, values, np.nan)
    line, = axis.plot(iteration, positive, linewidth=linewidth, label=label)
    negative = np.where(values < 0.0, -values, np.nan)
    if np.any(np.isfinite(negative)):
        axis.plot(iteration, negative, linewidth=linewidth, linestyle="--", color=line.get_color())


def apply_log_y_scale(axis: plt.Axes, values: np.ndarray) -> None:
    """Apply one log y-axis and matching limits to one log-derived plot axis."""
    axis.set_yscale("log")
    axis.set_ylim(*log_axis_limits(values))


def bool_token(value: bool) -> str:
    """Format one boolean in BATSRUS-style ``T/F`` text."""
    return "T" if value else "F"


def parse_session_settings(lines: list[str]) -> tuple[dict[str, bool], dict[str, str]]:
    """Parse tracked booleans and scalars from one PARAM.in session block."""
    bools: dict[str, bool] = {}
    scalars: dict[str, str] = {}
    current_section: str | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("!"):
            continue
        if stripped.startswith("#"):
            current_section = stripped.split()[0]
            continue

        parts = stripped.split()
        first = parts[0]

        if current_section in SECTION_BOOLEAN_NAMES and first in BOOL_TOKEN_TO_VALUE:
            bools[SECTION_BOOLEAN_NAMES[current_section]] = BOOL_TOKEN_TO_VALUE[first]
            current_section = None
            continue

        if len(parts) >= 2 and first in BOOL_TOKEN_TO_VALUE and IDENTIFIER_PATTERN.fullmatch(parts[1]):
            bools[parts[1]] = BOOL_TOKEN_TO_VALUE[first]

        if len(parts) >= 2 and parts[1] in TRACKED_SCALAR_NAMES:
            scalars[parts[1]] = parts[0]
        elif len(parts) >= 2 and parts[-1] in TRACKED_SCALAR_NAMES:
            scalars[parts[-1]] = " ".join(parts[:-1])

        current_section = None

    return bools, scalars


def summarize_active_session(
    active_bools: dict[str, bool],
    active_scalars: dict[str, str],
) -> str:
    """Build one compact session-state summary for plot labels."""
    amr = bool_token(active_bools.get("DoAmr", False))
    heat_flux = (
        f"{bool_token(active_bools.get('UseHeatFluxRegion', False))}/"
        f"{bool_token(active_bools.get('UseHeatFluxCollisionless', False))}"
    )
    zdi = (
        f"{bool_token(active_bools.get('UseZdiMagnetogram', False))}/"
        f"{bool_token(active_bools.get('UseZdiBoundary', False))}"
    )
    parts = [f"AMR={amr}", f"HF={heat_flux}", f"ZDI={zdi}"]
    ramp_start = active_scalars.get("ZdiRampIterStart")
    ramp_stop = active_scalars.get("ZdiRampIterStop")
    if active_bools.get("UseZdiBoundary", False) and ramp_start is not None and ramp_stop is not None:
        parts.append(f"Ramp={ramp_start}-{ramp_stop}")
    return "; ".join(parts)


def session_infos(run_root: Path | None, iteration_offset: int) -> list[SessionInfo]:
    """Parse session summaries from the run ``PARAM.in`` file."""
    if run_root is None:
        return []

    param_path = run_root / "PARAM.in"
    if not param_path.exists():
        return []

    raw_sessions: list[tuple[str, int, int, list[str]]] = []
    current_session_name: str | None = None
    current_session_lines: list[str] = []
    current_start = 0
    with param_path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("Begin session:"):
                current_session_name = f"session {stripped.split(':', 1)[1].strip()}"
                current_session_lines = []
                continue
            if current_session_name is None:
                continue
            current_session_lines.append(line.rstrip())
            if "MaxIteration" not in stripped:
                continue
            parts = stripped.split()
            try:
                stop_iteration = int(float(parts[0]))
            except (ValueError, IndexError):
                continue
            if stop_iteration < current_start:
                stop_iteration += current_start
            raw_sessions.append((current_session_name, current_start, stop_iteration, current_session_lines.copy()))
            current_start = stop_iteration
            current_session_name = None
            current_session_lines = []

    name_counts: dict[str, int] = {}
    for name, _, _, _ in raw_sessions:
        name_counts[name] = name_counts.get(name, 0) + 1

    name_seen: dict[str, int] = {}
    cumulative_bools: dict[str, bool] = {}
    cumulative_scalars: dict[str, str] = {}
    infos: list[SessionInfo] = []
    for name, start_local, stop_local, lines in raw_sessions:
        name_seen[name] = name_seen.get(name, 0) + 1
        if name_counts[name] == 1:
            label = name.replace("session ", "s")
        else:
            suffix = chr(ord("a") + name_seen[name] - 1)
            label = name.replace("session ", "s") + suffix
        local_bools, local_scalars = parse_session_settings(lines)
        cumulative_bools.update(local_bools)
        cumulative_scalars.update(local_scalars)
        infos.append(
            SessionInfo(
                label=label,
                start=iteration_offset + start_local,
                stop=iteration_offset + stop_local,
                local_bools=local_bools,
                local_scalars=local_scalars,
                active_bools=dict(cumulative_bools),
                active_scalars=dict(cumulative_scalars),
                summary=summarize_active_session(cumulative_bools, cumulative_scalars),
            )
        )
    return infos


def format_named_values(values: dict[str, bool | str]) -> str:
    """Format one mapping as ``key=value`` comma-separated text."""
    return ", ".join(f"{name}={values[name]}" for name in sorted(values))


def write_session_report(
    output_dir: Path,
    *,
    stem: str,
    log_name: str,
    segment_label: str,
    sessions: list[SessionInfo],
) -> Path | None:
    """Write one text summary of parsed PARAM.in sessions."""
    if not sessions:
        return None

    lines = [log_name, segment_label, ""]
    for info in sessions:
        lines.append(f"{info.label}: it {info.start}-{info.stop}")
        lines.append(f"state: {info.summary}")
        enabled_here = {name: bool_token(value) for name, value in info.local_bools.items() if value}
        disabled_here = {name: bool_token(value) for name, value in info.local_bools.items() if not value}
        if enabled_here:
            lines.append(f"enabled here: {format_named_values(enabled_here)}")
        if disabled_here:
            lines.append(f"disabled here: {format_named_values(disabled_here)}")
        if info.local_scalars:
            lines.append(f"set here: {format_named_values(info.local_scalars)}")
        active_flags = {name: bool_token(info.active_bools.get(name, False)) for name in REPORT_FLAG_NAMES}
        lines.append(f"active key flags: {format_named_values(active_flags)}")
        lines.append("")

    report_path = output_dir / f"{stem}.sessions.txt"
    report_path.write_text("\n".join(lines).rstrip() + "\n")
    return report_path


def zdi_ramp_ranges(sessions: list[SessionInfo]) -> list[tuple[int, int, str]]:
    """Return distinct absolute ZDI ramp ranges active in the parsed sessions."""
    ranges: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for info in sessions:
        if not (info.active_bools.get("UseZdiBoundary", False) or info.active_bools.get("UseZdiMagnetogram", False)):
            continue
        ramp_start_text = info.active_scalars.get("ZdiRampIterStart")
        ramp_stop_text = info.active_scalars.get("ZdiRampIterStop")
        if ramp_start_text is None or ramp_stop_text is None:
            continue
        ramp_start = resolve_session_iteration(ramp_start_text, session_start=info.start)
        ramp_stop = resolve_session_iteration(ramp_stop_text, session_start=info.start)
        if ramp_stop <= ramp_start:
            continue
        ramp_type = info.active_scalars.get("TypeZdiRamp", "ramp")
        ramp_info = (ramp_start, ramp_stop, ramp_type)
        if ramp_info in seen:
            continue
        seen.add(ramp_info)
        ranges.append(ramp_info)
    return ranges


def shade_session_ranges(axis: plt.Axes, iteration: np.ndarray, sessions: list[SessionInfo]) -> None:
    """Shade background ranges for the active run sessions."""
    for index, info in enumerate(sessions):
        overlap_start = max(info.start, iteration[0])
        overlap_stop = min(info.stop, iteration[-1])
        if overlap_start >= overlap_stop:
            continue
        axis.axvspan(overlap_start, overlap_stop, color=str(0.98 - 0.03 * (index % 2)), zorder=0)
        if iteration[0] <= info.stop <= iteration[-1]:
            axis.axvline(info.stop, color="0.4", linewidth=1.0, linestyle="--", alpha=0.5)


def shade_zdi_ramp_ranges(axis: plt.Axes, iteration: np.ndarray, sessions: list[SessionInfo]) -> None:
    """Highlight active ZDI ramp intervals with a strong shaded band."""
    for ramp_start, ramp_stop, _ramp_type in zdi_ramp_ranges(sessions):
        overlap_start = max(ramp_start, int(iteration[0]))
        overlap_stop = min(ramp_stop, int(iteration[-1]))
        if overlap_start >= overlap_stop:
            continue
        axis.axvspan(
            overlap_start,
            overlap_stop,
            facecolor="#ffb347",
            edgecolor="#c75b12",
            linewidth=1.2,
            alpha=0.35,
            zorder=0.5,
        )
        axis.axvline(overlap_start, color="#c75b12", linewidth=2.0, linestyle=":", alpha=0.95, zorder=0.6)
        axis.axvline(overlap_stop, color="#c75b12", linewidth=2.0, linestyle=":", alpha=0.95, zorder=0.6)


def label_session_ranges(axis: plt.Axes, iteration: np.ndarray, sessions: list[SessionInfo]) -> None:
    """Annotate session ranges on the top diagnostic panel."""
    for index, info in enumerate(sessions):
        overlap_start = max(info.start, iteration[0])
        overlap_stop = min(info.stop, iteration[-1])
        if overlap_start >= overlap_stop:
            continue
        axis.text(
            0.5 * (overlap_start + overlap_stop),
            0.98 if index % 2 == 0 else 0.90,
            f"{info.label}\n{info.summary}",
            transform=axis.get_xaxis_transform(),
            va="top",
            ha="center",
            fontsize=7,
            color="0.25",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.85},
        )


def label_zdi_ramp_ranges(axis: plt.Axes, iteration: np.ndarray, sessions: list[SessionInfo]) -> None:
    """Annotate highlighted ZDI ramp ranges on the top diagnostic panel."""
    for index, (ramp_start, ramp_stop, ramp_type) in enumerate(zdi_ramp_ranges(sessions)):
        overlap_start = max(ramp_start, int(iteration[0]))
        overlap_stop = min(ramp_stop, int(iteration[-1]))
        if overlap_start >= overlap_stop:
            continue
        axis.text(
            0.5 * (overlap_start + overlap_stop),
            0.79 if index % 2 == 0 else 0.71,
            f"ZDI {ramp_type} ramp\n{ramp_start}-{ramp_stop}",
            transform=axis.get_xaxis_transform(),
            va="top",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="#7a2e00",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ffe2b8", "edgecolor": "#c75b12", "alpha": 0.95},
        )


def plot_all_columns(
    *,
    output_dir: Path,
    stem: str,
    log_name: str,
    title: str,
    columns: list[str],
    data: np.ndarray,
    segment_label: str,
    sessions: list[SessionInfo],
) -> Path:
    """Plot every numeric log column in its own stacked panel."""
    iteration = data[:, 0]
    series_columns = list(enumerate(columns[1:], start=1))
    figure, axes = plt.subplots(
        len(series_columns),
        1,
        figsize=(12, max(2.0 * len(series_columns), 6.0)),
        sharex=True,
        constrained_layout=True,
    )
    if len(series_columns) == 1:
        axes = [axes]

    for axis, (column_index, column_name) in zip(axes, series_columns, strict=True):
        values = data[:, column_index]
        shade_session_ranges(axis, iteration, sessions)
        shade_zdi_ramp_ranges(axis, iteration, sessions)
        plot_signed_log_series(axis, iteration, values, linewidth=1.5)
        axis.set_ylabel(column_name)
        apply_log_y_scale(axis, values)
        axis.grid(True, alpha=0.3)

    axes[0].set_title("all columns")
    if sessions:
        label_session_ranges(axes[0], iteration, sessions)
        label_zdi_ramp_ranges(axes[0], iteration, sessions)
    axes[-1].set_xlabel(columns[0])
    for axis in axes:
        axis.set_xlim(iteration[0], iteration[-1])

    figure.suptitle(f"{title}\n{log_name}\n{segment_label}")
    output_path = output_dir / f"{stem}.all_columns.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def split_metric_name(name: str) -> tuple[str, str, str]:
    """Split one ``metric_R=radius`` log column into base, variant, and radius."""
    match = METRIC_COLUMN_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported metric column name: {name}")

    metric_name = match.group("metric")
    radius_label = match.group("radius")
    flux_match = FLUX_METRIC_PATTERN.fullmatch(metric_name)
    if flux_match is None:
        return metric_name, "value", radius_label

    base_name = flux_match.group("base")
    variant_name = flux_match.group("variant") or "total"
    return base_name, variant_name, radius_label


def radius_sort_key(label: str) -> tuple[int, object]:
    """Sort numeric radius labels numerically and everything else lexically."""
    try:
        return 0, float(label)
    except ValueError:
        return 1, label


def axis_specs(columns: list[str]) -> list[tuple[str, str, list[tuple[str, int]]]]:
    """Group radius-indexed log columns into plottable families."""
    groups: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for index, name in enumerate(columns):
        match = METRIC_COLUMN_PATTERN.fullmatch(name)
        if match is None:
            continue
        base_name, variant_name, radius_label = split_metric_name(name)
        groups.setdefault((base_name, variant_name), []).append((radius_label, index))

    specs: list[tuple[str, str, list[tuple[str, int]]]] = []
    for (base_name, variant_name), members in groups.items():
        specs.append(
            (
                base_name,
                variant_name,
                sorted(members, key=lambda item: radius_sort_key(item[0])),
            )
        )
    specs.sort(key=lambda item: (item[0], VARIANT_ORDER.get(item[1], 99)))
    return specs


def flux_panel_title(base_name: str, variant_name: str) -> str:
    """Return a human-readable title for one flux panel."""
    base_title = FLUX_DISPLAY_NAMES.get(base_name, base_name)
    if variant_name == "value":
        return base_title
    if variant_name == "total":
        return base_title
    if variant_name == "in":
        return f"{base_title} In"
    if variant_name == "out":
        return f"{base_title} Out"
    return f"{base_title} {variant_name}"


def corrected_panel_series(
    *,
    base_name: str,
    variant_name: str,
    members: list[tuple[str, int]],
    member_lookup: dict[tuple[str, str], list[tuple[str, int]]],
    data: np.ndarray,
) -> tuple[list[tuple[str, np.ndarray]], str]:
    """Return one plotted series group."""
    out = [(radius_label, np.array(data[:, column_index], copy=True)) for radius_label, column_index in members]
    title = flux_panel_title(base_name, variant_name)
    return out, title


def plot_flux_summary(
    *,
    output_dir: Path,
    stem: str,
    log_name: str,
    title: str,
    columns: list[str],
    data: np.ndarray,
    segment_label: str,
    sessions: list[SessionInfo],
) -> Path | None:
    """Plot stacked summaries of rho and jz shell flux logs, if present."""
    all_specs = axis_specs(columns)
    member_lookup = {(base_name, variant_name): members for base_name, variant_name, members in all_specs}
    selected_specs = [spec for spec in all_specs if (spec[0], spec[1]) in {("rho", "total"), ("jz", "total")}]
    if not selected_specs:
        return None

    iteration = data[:, 0]
    figure, axes = plt.subplots(
        len(selected_specs),
        1,
        figsize=(12, max(3.0 * len(selected_specs), 6.0)),
        sharex=True,
        constrained_layout=True,
    )
    if len(selected_specs) == 1:
        axes = [axes]

    for axis, (base_name, variant_name, members) in zip(axes, selected_specs, strict=True):
        shade_session_ranges(axis, iteration, sessions)
        shade_zdi_ramp_ranges(axis, iteration, sessions)
        plotted_members, axis_title = corrected_panel_series(
            base_name=base_name,
            variant_name=variant_name,
            members=members,
            member_lookup=member_lookup,
            data=data,
        )
        series_values: list[np.ndarray] = []
        for radius_label, values in plotted_members:
            series_values.append(values)
            plot_signed_log_series(axis, iteration, values, linewidth=2, label=f"R={radius_label}")
        axis.set_ylabel(axis_title)
        axis.set_title(axis_title)
        apply_log_y_scale(axis, np.concatenate(series_values))
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    if sessions:
        label_session_ranges(axes[0], iteration, sessions)
        label_zdi_ramp_ranges(axes[0], iteration, sessions)
    axes[-1].set_xlabel(columns[0])
    for axis in axes:
        axis.set_xlim(iteration[0], iteration[-1])

    figure.suptitle(f"{title}\n{log_name}\n{segment_label}")
    output_path = output_dir / f"{stem}.rho_jz.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def process_log_file(file_path: str | Path) -> None:
    """Process one BATSRUS log file into diagnostic plots and session text."""
    path = Path(file_path)
    if LOG_FILE_PATTERN.fullmatch(path.name) is None:
        raise ValueError(f"Unsupported log filename: {path.name}")

    output_dir = path.parent / "log"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("%s", path.name)
    title, columns, data = load_log_file(path)
    first_iteration = int(data[0, 0])
    segment_label = (
        f"restart segment starting at it={first_iteration}"
        if first_iteration > 0
        else "fresh segment starting at it=0"
    )
    run_root = find_run_root(path.parent)
    sessions = session_infos(run_root, first_iteration)
    stem = path.stem

    all_columns_path = plot_all_columns(
        output_dir=output_dir,
        stem=stem,
        log_name=path.name,
        title=title,
        columns=columns,
        data=data,
        segment_label=segment_label,
        sessions=sessions,
    )
    add_record("log_all_columns_png %r", str(all_columns_path.relative_to(path.parent)))

    flux_summary_path = plot_flux_summary(
        output_dir=output_dir,
        stem=stem,
        log_name=path.name,
        title=title,
        columns=columns,
        data=data,
        segment_label=segment_label,
        sessions=sessions,
    )
    if flux_summary_path is not None:
        add_record("log_rho_jz_png %r", str(flux_summary_path.relative_to(path.parent)))

    report_path = write_session_report(
        output_dir,
        stem=stem,
        log_name=path.name,
        segment_label=segment_label,
        sessions=sessions,
    )
    if report_path is not None:
        add_record("log_sessions_txt %r", str(report_path.relative_to(path.parent)))
