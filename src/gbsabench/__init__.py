"""gbsabench — shared helpers for the MM-GBSA affinity-ranking benchmark study.

Small, readable building blocks used by every notebook:
  paths    repo-relative locations (no absolute paths)
  style    the one navy/gold plotting style
  io       load the raw measurement CSVs
  metrics  the ranking/enrichment metric library (Kendall tau, BEDROC, ROC-AUC,
           DeLong, and the selection-corrected max-T permutation test)
"""

# Submodules are imported explicitly by callers (e.g. `from gbsabench import metrics`,
# `from gbsabench.paths import RAW`). We deliberately do NOT import them here so that
# `import gbsabench.paths` stays lightweight (no matplotlib/rdkit) and
# `python -m gbsabench.paths` runs without a runpy re-import warning.
__all__ = ["paths", "style", "io", "metrics"]
__version__ = "0.1.0"
