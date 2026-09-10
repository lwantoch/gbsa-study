"""discovery9 — shared helpers for the discovery-9 MD/BEDROC study.

Small, readable building blocks used by every notebook:
  paths    repo-relative locations (no absolute paths)
  style    the one navy/gold IDIS plotting style
  io       load per-complex features, GBSA scores, per-target BEDROC
  metrics  BEDROC α=k, rank-fusion helpers
"""
__all__ = ["paths", "style", "io", "metrics"]
__version__ = "0.1.0"
