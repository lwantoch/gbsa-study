#!/usr/bin/env python3
"""
Patch every notebook so figures actually get exported.

The problem: many notebooks use `plt.get_fignums()` in the last cell to save
figures, but by the time that cell runs the inline backend has already closed
every figure. `plt.get_fignums()` returns [], and 0 figures are saved.

Fix: install a `pyplot.figure` / `pyplot.subplots` monkeypatch in the preamble
that appends every created figure to `_SAVED_FIGS`. Then rewrite the export
cell to iterate `_SAVED_FIGS` (with a get_fignums fallback for anything the
monkeypatch missed).

This is idempotent - running it twice does nothing new.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO / "notebooks"

CAPTURE_MARKER = "# --- fig-capture hook (iter-3 fix) ---"
CAPTURE_SNIPPET = f"""
{CAPTURE_MARKER}
_SAVED_FIGS = globals().setdefault('_SAVED_FIGS', [])
_orig_figure = plt.figure
_orig_subplots = plt.subplots
def _figure_capture(*a, **kw):
    fig = _orig_figure(*a, **kw)
    if fig not in _SAVED_FIGS:
        _SAVED_FIGS.append(fig)
    return fig
def _subplots_capture(*a, **kw):
    fig, ax = _orig_subplots(*a, **kw)
    if fig not in _SAVED_FIGS:
        _SAVED_FIGS.append(fig)
    return fig, ax
plt.figure = _figure_capture
plt.subplots = _subplots_capture
"""

EXPORT_SNIPPET = """# --- export every figure produced in this notebook (iter-3 fix) ---
try:
    FIGURES.mkdir(parents=True, exist_ok=True)
except NameError:
    from discovery9.paths import FIGURES
    FIGURES.mkdir(parents=True, exist_ok=True)
try:
    _cream = CREAM
except NameError:
    from discovery9.style import CREAM as _cream
figs = list(globals().get('_SAVED_FIGS', []))
# fallback: any figures still open in the backend
for num in plt.get_fignums():
    f = plt.figure(num)
    if f not in figs:
        figs.append(f)
saved = []
for i, fig in enumerate(figs, start=1):
    out = FIGURES / f"{NB_STEM}_fig{i}.png"
    try:
        fig.savefig(out, bbox_inches='tight', dpi=140, facecolor=_cream)
    except Exception as e:
        print(f'  WARN: failed to save fig{i}: {e}')
        continue
    saved.append(str(out.name))
print(f'saved {len(saved)} figures:')
for s in saved:
    print(' ', s)
"""


def _cell_source(cell) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _set_source(cell, text: str) -> None:
    cell["source"] = text
    cell["outputs"] = []
    cell["execution_count"] = None


def patch_notebook(path: Path) -> tuple[bool, str]:
    with open(path) as fh:
        nb = json.load(fh)

    changed = False
    notes = []

    # 1. Inject the capture hook into the preamble cell (first code cell that
    # imports matplotlib).
    preamble_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = _cell_source(cell)
        if "import matplotlib.pyplot as plt" in src:
            preamble_idx = i
            break
    if preamble_idx is None:
        return False, "no matplotlib preamble found"

    pre_src = _cell_source(nb["cells"][preamble_idx])
    if CAPTURE_MARKER not in pre_src:
        # Append capture snippet just after apply_style() call if present, else
        # at end of cell.
        if "apply_style()" in pre_src:
            new_src = pre_src.replace(
                "apply_style()",
                "apply_style()\n" + CAPTURE_SNIPPET,
                1,
            )
        else:
            new_src = pre_src.rstrip() + "\n" + CAPTURE_SNIPPET
        _set_source(nb["cells"][preamble_idx], new_src)
        changed = True
        notes.append("preamble patched")

    # 2. Replace the export cell (any code cell that already contains the
    # export pattern). Only one such cell is expected per notebook.
    export_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = _cell_source(cell)
        if ("get_fignums" in src and "savefig" in src) or (
            "_SAVED_FIGS" in src and "savefig" in src
        ):
            export_idx = i
            break

    if export_idx is not None:
        _set_source(nb["cells"][export_idx], EXPORT_SNIPPET)
        changed = True
        notes.append(f"export cell {export_idx} rewritten")
    else:
        notes.append("no recognizable export cell (skipped)")

    if changed:
        with open(path, "w") as fh:
            json.dump(nb, fh, indent=1)
    return changed, "; ".join(notes)


def main():
    nbs = sorted(NOTEBOOKS.glob("*.ipynb"))
    print(f"Patching {len(nbs)} notebooks:")
    for p in nbs:
        changed, note = patch_notebook(p)
        flag = "PATCHED" if changed else "unchanged"
        print(f"  [{flag}] {p.name}: {note}")


if __name__ == "__main__":
    main()
