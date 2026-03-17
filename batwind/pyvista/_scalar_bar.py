from __future__ import annotations


def readable_scalar_bar_args(title: str, *, extra: dict | None = None) -> dict[str, object]:
    args = {
        "title": title,
        "title_font_size": 36,
        "label_font_size": 30,
        "unconstrained_font_size": True,
    }
    if extra is not None:
        args.update(extra)
    return args


__all__ = ["readable_scalar_bar_args"]
