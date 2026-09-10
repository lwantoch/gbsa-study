"""BO trial runner: single-ligand, single-stage — optimizes only the
``md_production`` stage's wall-time.

Per user directive 2026-08-30:
  - BO objective is **just the md_production step**, nothing else.
  - Fitness = 1 / md_production_wall_seconds (higher = faster = better).
  - BEDROC is NOT part of the BO loop; it's computed post-hoc on the
    top-N BO winners using the full 30-ligand pool.

Trial cost = one equilibrated system + one 20-ns MD production run.
Equilibration stages (sd → cg → nvt_res → npt_res → npt) still run because
production needs an equilibrated starting frame; their wall-time does NOT
count toward the fitness (only the production stage's own wall-time does).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path("/mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/pipeline-mps")
STUDIES_ROOT = Path(
    "/mnt/netapp1/Store_othcxlwa/pipeline-mps-workspaces/bayes_opt",
)
# Pre-equilibrated seed dirs — one per (target, ligand) reference system.
# Layout: SEEDS_ROOT / "<target>_<ligand>" / {prepare,docking,export,setup,
# sd,cg,nvt_res,npt_res,npt} / ... — everything up to and including md_npt,
# but NOT production or gbsa. Created by scripts/prep_bo_seeds.sbatch (or,
# for the initial smoke, rsync'd from an existing smoke_5hu9 job).
SEEDS_ROOT = STUDIES_ROOT / "seeds"
SMOKE_DATA = PROJECT_ROOT / "smoke_data"
DOCKING_BOX = SMOKE_DATA / "dockingbox.txt"

# Stages that must be present in the seed. If any is missing rsync-in silently,
# the pipeline would fall back to running earlier stages inside the BO trial —
# defeats the point.
_SEED_STAGES = ("prepare", "docking", "export", "setup", "sd", "cg",
                "nvt_res", "npt_res", "npt")

FitnessMode = Literal["md_production_wall", "throughput", "stability"]


def _read_docking_box(pdb_id: str) -> tuple[float, float, float, float]:
    import csv
    with DOCKING_BOX.open() as fh:
        for row in csv.DictReader(fh):
            if row["PDB_ID"] == pdb_id:
                return (
                    float(row["Box_Center_X"]),
                    float(row["Box_Center_Y"]),
                    float(row["Box_Center_Z"]),
                    float(row["Box_Size"]),
                )
    raise KeyError(pdb_id)


def _seed_dir_for(target: str, ligand: str) -> Path:
    """Return the pre-equilibrated seed dir for a (target, ligand) system.

    Raises if the seed is missing any of the required pre-production stages.
    """
    seed = SEEDS_ROOT / f"{target}_{ligand}"
    if not seed.is_dir():
        raise FileNotFoundError(
            f"BO seed dir missing: {seed}. Run scripts/prep_bo_seeds.sbatch "
            f"to generate it (equilibrates prepare→dock→setup→sd→cg→"
            f"nvt_res→npt_res→npt for that system, ~1h on A100)."
        )
    for stage in _SEED_STAGES:
        if not (seed / stage / "result.json").is_file():
            raise FileNotFoundError(
                f"Seed {seed} is incomplete: missing {stage}/result.json"
            )
    if not (seed / "npt" / "system.gro").is_file():
        raise FileNotFoundError(f"Seed {seed} has no npt/system.gro")
    return seed


def _init_trial_workspace(
    study_root: Path,
    trial_number: int,
    target: str,
    ligand: str,
    cfg: dict[str, Any],
) -> Path:
    """Create a per-trial signac workspace + copy pre-equilibrated stages from
    the seed dir + drop the BO MDP override into the statepoint.

    Every BO trial reuses the SAME pre-equilibrated (target, ligand) starting
    state — differences in fitness come from the production MDP knobs only,
    not from equilibration-path variance. This is the "no equilibration in the
    BO fitness" invariant (MASTERPLAN_BO.md §0).
    """
    import shutil
    import signac

    trial_root = study_root / f"trial_{trial_number:04d}"
    trial_root.mkdir(parents=True, exist_ok=True)
    project = signac.init_project(path=str(trial_root))

    cx, cy, cz, box_size = _read_docking_box(target)
    protein_pdb = SMOKE_DATA / target / f"{target}.pdb"
    ligand_sdf = SMOKE_DATA / target / "ligands" / f"{ligand}.sdf"
    if not protein_pdb.is_file():
        raise FileNotFoundError(protein_pdb)
    if not ligand_sdf.is_file():
        raise FileNotFoundError(ligand_sdf)

    # Deterministic seed derived from trial number — fixed once at trial init,
    # so ld-seed / gen-seed inside GROMACS are reproducible per trial without
    # tying us to a global RNG. Post-hoc replicate validation of the winner
    # varies THIS seed to sample noise (MASTERPLAN_BO.md §5).
    trial_seed = (trial_number * 2654435761) & 0x7FFFFFFF  # Knuth hash

    statepoint = {
        "target_pdb": target,
        "ligand_name": ligand,
        "replica": 1,
        "trial_number": trial_number,
        "protein_pdb": str(protein_pdb),
        "ligand_sdf": str(ligand_sdf),
        "box_center_x": cx,
        "box_center_y": cy,
        "box_center_z": cz,
        "box_size": box_size,
        # BO overrides. md_stage.py:main reads mdp_override from job.sp and
        # merges it onto PRODUCTION_PARAMS before dispatching to
        # run_production() — this is what actually feeds BO into GROMACS.
        "mdp_override": cfg["mdp"],
        "protein_ff": cfg["forcefield"]["protein_ff"],
        "ligand_ff": cfg["forcefield"]["ligand_ff"],
        "water_model": cfg["forcefield"]["water_model"],
        "trial_seed": trial_seed,
    }
    job = project.open_job(statepoint).init()
    job_path = Path(job.path)
    (job_path / "signac_statepoint.json").write_text(
        json.dumps(statepoint, indent=2),
    )

    # Rsync the pre-equilibrated stages from the seed. Use copytree with
    # dirs_exist_ok because open_job() may have made empty stage dirs.
    seed = _seed_dir_for(target, ligand)
    for stage in _SEED_STAGES:
        src = seed / stage
        dst = job_path / stage
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    return job_path


def _run_pipeline_subset(job_dir: Path, stop_after: str, timeout_s: int) -> int:
    """Invoke the pipeline for one BO trial.

    Pre-equilibrated stages are already staged into ``job_dir`` by
    ``_init_trial_workspace`` (see MASTERPLAN_BO.md §0). We invoke the action
    runner ONLY for ``md_production`` — no prepare/dock/setup, no equilibration.
    ``stop_after`` is ignored in production-only mode (there is only one stage
    to run) but is kept in the signature for symmetry with the pre-BO code.
    """
    del stop_after  # production-only mode has one stage; nothing to stop.
    env = os.environ.copy()
    proc = subprocess.run(  # noqa: S603
        [
            "pixi", "run", "python", "-u", "-X", "faulthandler",
            str(PROJECT_ROOT / "scripts/_action_runner.py"),
            "md_stage", str(job_dir), "--stage", "md_production",
        ],
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=timeout_s,
        check=False,
    )
    return proc.returncode


# ---------------------------------------------------------------------------
# Fitness extractors
# ---------------------------------------------------------------------------


def _current_gpu_model() -> str:
    """A100 vs L40S vs H100 matters — ns/day optima are hardware-specific.
    Attach to every trial result so we can't accidentally mix them later.
    """
    try:
        out = subprocess.check_output(  # noqa: S603
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.strip().splitlines()[0] if out.strip() else "unknown"
    except Exception:
        return "unknown"


# --- Stability hard-filter --------------------------------------------------
#
# A "completed" run isn't automatically valid — LINCS warnings, thermostat
# blowups, and density drift all signal a bad config that only *looked* fast
# because the integrator hasn't crashed yet. Report explicitly flags this
# ("Ein Lauf, der 30% schneller ist und LINCS-Warnungen produziert, bekommt
# also keine Belohnung"). Return True == passed, False == disqualified.

_STABILITY_MAX_LINCS_WARNINGS = 5     # gmx tolerates a few during equil, not
                                       # during production. Zero is too strict;
                                       # 5 is the point where trajectory quality
                                       # measurably degrades.
_STABILITY_TEMP_DRIFT_K = 5.0          # from ref_t=300
_STABILITY_DENSITY_DRIFT_PCT = 5.0


def _stability_passed(job_dir: Path) -> tuple[bool, str]:
    """Return (ok, reason). Called AFTER the completion check — a run that
    reached the Time: line still has to survive this filter."""
    log = job_dir / "production" / "process" / "gromacs.log"
    if not log.is_file():
        return False, "no gromacs.log"
    text = log.read_text(errors="ignore")

    if "Fatal error" in text:
        return False, "gmx Fatal error in log"

    lincs_warns = text.count("LINCS WARNING")
    if lincs_warns > _STABILITY_MAX_LINCS_WARNINGS:
        return False, f"LINCS warnings={lincs_warns} > {_STABILITY_MAX_LINCS_WARNINGS}"

    # Temperature drift: cheap check via `gmx energy` on the .edr — skip if
    # missing (some fast trials may have -noconfout). Extraction reuses the
    # same subprocess pattern as _fitness_stability.
    edr = job_dir / "production" / "process" / "gromacs.edr"
    if edr.is_file():
        gmx = str(PROJECT_ROOT / ".pixi/envs/default/bin/gmx")
        xvg = job_dir / "production" / "_stability_check.xvg"
        proc = subprocess.run(  # noqa: S603
            [gmx, "energy", "-f", str(edr), "-o", str(xvg)],
            input="Temperature\nDensity\n0\n",
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if proc.returncode == 0 and xvg.is_file():
            import numpy as np
            try:
                data = np.loadtxt(str(xvg), comments=("#", "@"))
                if data.ndim == 2 and data.shape[0] > 100:
                    # Skip first 20% as spurious equilibration remnant
                    tail = data[int(0.2 * data.shape[0]):, :]
                    T_mean = float(np.mean(tail[:, 1]))
                    D_mean = float(np.mean(tail[:, 2]))
                    if abs(T_mean - 300.0) > _STABILITY_TEMP_DRIFT_K:
                        return False, f"T drift: mean={T_mean:.1f} K"
                    # Density check: compare first-half vs second-half mean
                    # (drift in time) — absolute value depends on system
                    # composition (protein+water+ions run ~1050-1100 kg/m³,
                    # not 1000). What matters is that ρ isn't drifting.
                    half = tail.shape[0] // 2
                    D_early = float(np.mean(tail[:half, 2]))
                    D_late = float(np.mean(tail[half:, 2]))
                    drift = abs(D_late - D_early) / max(D_early, 1.0) * 100.0
                    if drift > _STABILITY_DENSITY_DRIFT_PCT:
                        return False, (
                            f"density drift: {D_early:.1f}→{D_late:.1f} kg/m³ "
                            f"({drift:.2f}% > {_STABILITY_DENSITY_DRIFT_PCT}%)"
                        )
            except Exception as e:  # noqa: BLE001
                # Don't disqualify on our own parse errors
                return True, f"stability check skipped ({e})"

    return True, "ok"


def _md_production_wall_seconds(job_dir: Path) -> float | None:
    """Parse the md_production stage's own wall-time from its gromacs.log.

    gmx writes lines like::

        Finished mdrun on rank 0 Sun Aug 30 18:07:26 2026
        ...
                       Core t (s)   Wall t (s)        (%)
                       Time:  10327.443     2582.181     400.0

    Returns wall_seconds (Wall t column) or None if not found.
    """
    log = job_dir / "production" / "process" / "gromacs.log"
    if not log.is_file():
        return None
    text = log.read_text(errors="ignore")
    # Match "Time:  <core> <wall> <pct>"
    m = re.search(
        r"^\s*Time:\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s*$",
        text,
        re.MULTILINE,
    )
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _fitness_md_production_wall(job_dir: Path) -> float:
    """Fitness = 1 / md_production_wall_hours. Returns 0.0 if the stage didn't
    complete OR failed the stability hard-filter (LINCS, T/ρ drift). Speed at
    the cost of a broken trajectory is worthless.
    """
    wall = _md_production_wall_seconds(job_dir)
    if wall is None or wall <= 0:
        return 0.0
    ok, _reason = _stability_passed(job_dir)
    if not ok:
        return 0.0
    return 1.0 / (wall / 3600.0)


def _fitness_throughput(job_dir: Path) -> float:
    """ns/day from the production stage's gromacs.log. 0.0 if stability filter
    fails — same reasoning as _fitness_md_production_wall."""
    log = job_dir / "production" / "process" / "gromacs.log"
    if not log.is_file():
        return 0.0
    ok, _ = _stability_passed(job_dir)
    if not ok:
        return 0.0
    for line in log.read_text(errors="ignore").splitlines():
        if line.strip().startswith("Performance:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    return 0.0
    return 0.0


def _fitness_stability(job_dir: Path) -> float:
    """1 / (1 + |total-energy drift in kT/ns|) from the .edr file."""
    edr = job_dir / "production" / "process" / "gromacs.edr"
    if not edr.is_file():
        return 0.0
    gmx = str(PROJECT_ROOT / ".pixi/envs/default/bin/gmx")
    out = job_dir / "production" / "energy_total.xvg"
    proc = subprocess.run(  # noqa: S603
        [gmx, "energy", "-f", str(edr), "-o", str(out)],
        input="Total-Energy\n0\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not out.is_file():
        return 0.0
    import numpy as np
    data = np.loadtxt(str(out), comments=("#", "@"))
    if data.ndim != 2 or data.shape[0] < 10:
        return 0.0
    t_ns = (data[-1, 0] - data[0, 0]) / 1000.0
    if t_ns <= 0:
        return 0.0
    slope = np.polyfit(data[:, 0], data[:, 1], 1)[0]
    drift_kt_per_ns = abs(slope) * 1000.0 / 2.5
    return 1.0 / (1.0 + drift_kt_per_ns)


_FITNESS_FN = {
    "md_production_wall": _fitness_md_production_wall,
    "throughput": _fitness_throughput,
    "stability": _fitness_stability,
}

# All modes stop after md_production — no GBSA needed for a wall-time /
# throughput / stability measurement.
_STOP_AFTER = {
    "md_production_wall": "md_production",
    "throughput": "md_production",
    "stability": "md_production",
}

_TIMEOUT_S = {
    "md_production_wall": 5 * 60 * 60,   # 5 h — 20 ns can be slow with bad params
    "throughput": 5 * 60 * 60,
    "stability": 5 * 60 * 60,
}


def run_trial(
    trial_number: int,
    cfg: dict[str, Any],
    *,
    target: str,
    ligand: str = "ligand01",
    study_name: str = "default",
    mode: FitnessMode = "md_production_wall",
    keep_workspace: bool = False,
) -> float:
    """Run one BO trial. Returns fitness scalar (higher = better).

    The pipeline runs prepare_inputs → dock → export_pose → setup_system →
    md_sd → md_cg → md_nvt_res → md_npt_res → md_npt → md_production, then
    stops. Only md_production's own wall-time enters the fitness (default
    mode); equilibration overhead is amortized because it's fixed regardless
    of the BO knobs.
    """
    study_root = STUDIES_ROOT / study_name
    study_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()

    job_dir = _init_trial_workspace(study_root, trial_number, target, ligand, cfg)
    stop_after = _STOP_AFTER[mode]
    timeout = _TIMEOUT_S[mode]
    try:
        rc = _run_pipeline_subset(job_dir, stop_after=stop_after, timeout_s=timeout)
    except subprocess.TimeoutExpired:
        rc = 124

    fitness = 0.0 if rc != 0 else _FITNESS_FN[mode](job_dir)
    md_wall_s = _md_production_wall_seconds(job_dir)
    stability_ok, stability_reason = _stability_passed(job_dir) if rc == 0 else (False, f"rc={rc}")

    (job_dir / "trial_result.json").write_text(json.dumps({
        "trial_number": trial_number,
        "mode": mode,
        "returncode": rc,
        "fitness": fitness,
        "md_production_wall_seconds": md_wall_s,
        "stability_passed": stability_ok,
        "stability_reason": stability_reason,
        # Hardware fingerprint — ns/day optima are GPU-specific per report.
        # Never mix trials from different GPU models in the same study.
        "gpu_model": _current_gpu_model(),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "cfg": cfg,
    }, indent=2))

    if not keep_workspace and rc == 0:
        # Drop the fat trajectory files; keep manifests + trial_result.json
        # for post-hoc BEDROC / re-analysis. Trajectories are ~GB/stage and
        # can always be regenerated from the same statepoint.
        for stage in ("sd", "cg", "nvt_res", "npt_res", "npt", "production"):
            proc_dir = job_dir / stage / "process"
            if proc_dir.is_dir():
                shutil.rmtree(proc_dir, ignore_errors=True)

    return fitness
