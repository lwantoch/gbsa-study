#!/usr/bin/env python3
"""Bundle SI PNGs into PDFs.

Produces:
  figures/SI/pdf/discovery9_SI_all.pdf   — one master PDF (72 pages), ordered by target then analysis
  figures/SI/pdf/{TARGET}.pdf            — 9 per-target PDFs, 8 pages each
  figures/SI/pdf/by_analysis/{NN}_{stem}.pdf — 8 per-analysis PDFs, 9 pages each (one target per page)
"""
from pathlib import Path
from PIL import Image

ROOT = Path('/mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/gbsa-study')
SI = ROOT / 'figures' / 'SI'
OUT = SI / 'pdf'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'by_analysis').mkdir(parents=True, exist_ok=True)

# collect all per-target subdirs; each has 8 PNGs 01_*.png … 08_*.png
targets = sorted([d.name for d in SI.iterdir() if d.is_dir() and d.name != 'pdf'])
print(f'{len(targets)} target dirs: {targets}')

def to_rgb(path: Path) -> Image.Image:
    im = Image.open(path).convert('RGB')
    return im

# 1. Per-target PDF
per_target_pages = {}
for t in targets:
    pngs = sorted((SI / t).glob('[0-9]*.png'))
    if not pngs:
        continue
    ims = [to_rgb(p) for p in pngs]
    out = OUT / f'{t}.pdf'
    ims[0].save(out, save_all=True, append_images=ims[1:], resolution=200.0)
    per_target_pages[t] = pngs
    print(f'  wrote {out.name}   ({len(ims)} pages)')

# 2. Per-analysis PDF (analysis number → one PDF with one page per target)
by_num = {}
for t in targets:
    for p in sorted((SI / t).glob('[0-9]*.png')):
        num_stem = p.name.split('.')[0]  # e.g. "01_ligand_drift"
        by_num.setdefault(num_stem, []).append((t, p))
for num_stem, entries in sorted(by_num.items()):
    ims = [to_rgb(p) for _, p in entries]
    out = OUT / 'by_analysis' / f'{num_stem}.pdf'
    ims[0].save(out, save_all=True, append_images=ims[1:], resolution=200.0)
    print(f'  wrote by_analysis/{out.name}   ({len(ims)} pages)')

# 3. Master PDF: target-major (all 8 analyses of one target, then next target)
all_pngs = []
for t in targets:
    all_pngs.extend(sorted((SI / t).glob('[0-9]*.png')))
if all_pngs:
    ims = [to_rgb(p) for p in all_pngs]
    out = OUT / 'discovery9_SI_all.pdf'
    ims[0].save(out, save_all=True, append_images=ims[1:], resolution=200.0)
    print(f'\nWROTE MASTER: {out.name}   ({len(ims)} pages)')

print(f'\nPDFs in: {OUT}/')
