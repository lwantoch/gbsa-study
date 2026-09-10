#!/usr/bin/env python3
"""One-shot splitter that turns the old monolithic analysis.ipynb into 17 numbered
per-experiment notebooks under notebooks/, each self-contained.

Each new notebook gets:
  - a NB_STEM header cell (so figures export as {NB_STEM}_figK.png)
  - a preamble cell that imports the src.discovery9 helpers and calls apply_style()
  - the original markdown + code cells for that section
  - a final export cell that saves every open figure to FIGURES/

The auto-preamble `try: _ = df; except NameError: raise` guards in the monolith
are stripped (the new preamble does the loading properly).

Run once after the src/ package + directory tree are in place.
"""
from __future__ import annotations
import json, re, uuid, shutil
from pathlib import Path

ROOT = Path('/mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/gbsa-study')
MONOLITH = ROOT / 'analysis.ipynb'
NB_OUT = ROOT / 'notebooks'
NB_OUT.mkdir(parents=True, exist_ok=True)

# Monolith §N.X-header  →  new notebook stem
# Some new notebooks bundle two monolith sections; some monolith sections split into two.
MAPPING = [
    # (new_stem, [list of monolith '## N.' prefixes to include], one-line description)
    ('01_dataset_overview',        ['## 1. '], 'Per-target counts, actives/decoys balance, protein-level invariants.'),
    ('02_per_complex_deep_dive',   ['## 2. '], 'One complex: 8-panel time series, HB, contacts, RMSF.'),
    ('03_per_target_aggregation',  ['## 3. '], 'Per-target means + min-max scaled heatmap.'),
    ('04_feature_distributions',   ['## 4. '], 'Feature distributions actives vs decoys per target.'),
    ('05_kinematic_phase_space',   ['## 5. '], 'Ligand drift vs COM displacement per target.'),
    ('06_time_series_overlay',     ['## 6. '], 'All 30 ligand drift trajectories overlaid per target.'),
    ('07_active_site_fingerprint', ['## 7. '], 'Residue × complex contact-persistence heatmap.'),
    ('08_feature_correlation',     ['## 8. '], 'Within-target Pearson r heatmap.'),
    ('09_actives_decoys_separation', ['## 9. '], 'z(actives) − z(decoys) per feature per target.'),
    ('10_bedroc_baselines',        ['## 10. ', '## 11. '], 'Docking vs GBSA vs MD composite BEDROC per target.'),
    ('11_ml_combo_selection',      ['## 12. ', '## 13. '], 'LOTO regressors pick per-target best sp-config (result: no signal).'),
    ('12_full_factorial_ml',       ['## 14. '], 'MD × combo × GBSA meta-model (result: no lift).'),
    ('13_deep_research_verdict',   ['## 15. '], 'Honest summary of what ML does and does not do here.'),
    ('14_gbsa_param_vs_md',        ['## 16. '], 'MD signatures vs GBSA parameter preferences (Spearman + BH-FDR).'),
    ('15_ligand_chem_gbsa_surrogate', ['## 17. '], 'Real ligand chemistry + ML surrogate for GBSA ΔG.'),
    ('16_single_feature_bedroc',   ['## 18. '], 'Do individual MD components beat GBSA? Yes: lig_buried_sasa_std_A2.'),
    ('17_rank_fusion_deployable',  ['## 19. '], 'LOTO K=1 rank-fusion — the one deployable positive result.'),
]

# Strip the monolith's "safety guard" preamble that references cell 0 setup
GUARD_RE = re.compile(
    r"try:.*?except NameError:.*?raise RuntimeError\('Please run.*?first\.?'\)\n",
    flags=re.DOTALL,
)

def clean_code_source(src: str) -> str:
    # Remove the safety guard block
    src2 = GUARD_RE.sub('', src)
    # Also drop the top comment "# --- auto-preamble (safe to skip if you ran cell above) ---"
    src2 = re.sub(r'^# --- auto-preamble.*?---\n', '', src2, flags=re.MULTILINE)
    return src2

def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:12], "metadata": {},
            "source": text.splitlines(keepends=True)}

def code_cell(text: str) -> dict:
    return {"cell_type": "code", "id": uuid.uuid4().hex[:12], "execution_count": None,
            "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}

PREAMBLE_TEMPLATE = """# --- notebook preamble ---
NB_STEM = "{stem}"

import sys, os, json, glob
from pathlib import Path

# Make the in-repo src package importable without an install
sys.path.insert(0, str(Path(os.path.abspath('..')) / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from discovery9.style import apply_style, NAVY, GOLD, GREY, GREY_DASH as GREYD, CREAM, WHITE, ACTIVE, DECOY, WARN
from discovery9.paths import ROOT, RAW, DERIVED, EXTERNAL, FIGURES, TABLES, GBSA_STUDY
from discovery9.io    import load_features, load_gbsa, load_gbsa_all, load_metadata, load_bedroc_matrix, load_bedroc_all_combos, load_per_complex_analysis
from discovery9.metrics import bedroc, bedroc_per_target, rank_fuse
apply_style()

# Default: load the master feature table (with ligand-chem descriptors when available)
df = load_features(with_ligand_chem=True)
print(f'features.parquet: {{len(df)}} complexes × {{df.shape[1]}} columns  ·  targets: {{df.target.nunique()}}')
"""

EXPORT_TEMPLATE = """# --- export every figure produced in this notebook ---
FIGURES.mkdir(parents=True, exist_ok=True)
saved = []
for i, num in enumerate(plt.get_fignums(), start=1):
    out = FIGURES / f"{NB_STEM}_fig{i}.png"
    plt.figure(num).savefig(out, bbox_inches='tight', dpi=140, facecolor=CREAM)
    saved.append(str(out.relative_to(ROOT)))
print(f'saved {len(saved)} figures:')
for s in saved: print(' ', s)
"""
# NB: this template is not run through .format(); braces are literal Python f-string braces.

# Load monolith
nb = json.load(open(MONOLITH))
cells = nb['cells']

def cell_text(c):
    return ''.join(c['source']) if isinstance(c['source'], list) else c['source']

def section_index(cells):
    """Return list of (start_index, header_prefix) for each '## N. ' cell."""
    out = []
    for i, c in enumerate(cells):
        if c['cell_type'] != 'markdown': continue
        s = cell_text(c)
        m = re.match(r'^## (\d+)\. ', s.lstrip())
        if m:
            out.append((i, f'## {m.group(1)}. '))
    return out

sections = section_index(cells)
starts = {prefix: idx for idx, prefix in sections}
next_prefix = {}
for k, (idx, prefix) in enumerate(sections):
    next_prefix[prefix] = sections[k+1][1] if k+1 < len(sections) else None

def cells_for(prefix):
    start = starts.get(prefix)
    if start is None:
        return []
    nprefix = next_prefix[prefix]
    end = starts.get(nprefix, len(cells)) if nprefix else len(cells)
    return cells[start:end]

for new_stem, prefixes, desc in MAPPING:
    body = []
    for pref in prefixes:
        body.extend(cells_for(pref))
    if not body:
        print(f'WARN {new_stem}: no monolith cells matched prefixes {prefixes}')
        continue

    # Clean code cells
    clean_body = []
    for c in body:
        c2 = {'cell_type': c['cell_type'], 'id': uuid.uuid4().hex[:12],
              'metadata': c.get('metadata', {}),
              'source': c['source']}
        if c['cell_type'] == 'code':
            c2['outputs'] = []; c2['execution_count'] = None
            src = clean_code_source(cell_text(c))
            c2['source'] = src.splitlines(keepends=True)
        clean_body.append(c2)

    # Build the new notebook
    new_cells = [
        md_cell(f"# {new_stem}\n\n{desc}\n\n_(Notebook auto-generated by `reproduce/split_monolith.py`. Self-contained: loads its data via `discovery9.io`, exports figures to `figures/{new_stem}_figK.png`.)_"),
        code_cell(PREAMBLE_TEMPLATE.format(stem=new_stem)),
        *clean_body,
        code_cell(EXPORT_TEMPLATE),
    ]
    out_nb = {
        'cells': new_cells,
        'metadata': {
            'kernelspec': {'display_name': 'Discovery-9 (pixi)', 'language': 'python', 'name': 'discovery9'},
            'language_info': {'name': 'python', 'version': '3.11'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    out_path = NB_OUT / f'{new_stem}.ipynb'
    json.dump(out_nb, open(out_path, 'w'), indent=1)
    print(f'  {new_stem}.ipynb — {len(clean_body)} content cells + preamble/export')

print(f'\nWrote {len(MAPPING)} notebooks to {NB_OUT}/')
