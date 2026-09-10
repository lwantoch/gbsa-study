"""BO v2 search space — adds HMR, MTS, extended dt, LINCS-iter/order.

Written 2026-09-01 after v1 (`space.py`) capped `dt` at 2 fs and disabled
HMR / MTS out of caution. User directive: *"if there are errors we report the
errors, but skip nothing"* → we re-open the whole envelope and let grompp /
mdrun be the ground truth on what actually works. Failed trials record their
`rc` and `stability_reason` in `trial_result.json`; nothing is preemptively
gated out on empirical guesses about what "should" fail.

Hard invariants (documented grompp-fatal by construction, GROMACS 2025)
are still enforced by predicates — those are definitional impossibilities,
not empirical guesses:

  - integrator=md-vv  + pcoupl=Parrinello-Rahman           → grompp fatal
  - mts=True          + integrator != md                    → grompp fatal
  - mts=True          + constraints in {none}               → grompp fatal
  - pcoupl=Parrinello-Rahman + tcoupl=no                    → grompp fatal
  - sd integrator                                           → grompp forces tcoupl=no

Newly *released* axes (previously disabled by empirical caution):

  - dt        ∈ {0.001, 0.002, 0.003, 0.004, 0.005}
  - hmr       ∈ {1.0, 2.5, 3.0, 3.5, 4.0}
  - mts       ∈ {False, True}
  - mts_factor ∈ {2, 3, 4}   (only when mts=True; grompp key `mts-level2-factor`)
  - lincs_order ∈ {4, 6, 8}
  - lincs_iter  ∈ {1, 2}

The (dt, HMR, constraints) sub-space is the physically coupled one:
- dt ≥ 4 fs is only stable with HMR ≥ 3.0 (Hopkins et al. JCTC 2015) — but we
  *don't* enforce that; TPE will learn it because grompp/mdrun will crash
  every trial that violates it, and those crashes are the signal we want.

MTS block emits the GROMACS-2025 key set (see mdp.py: `mts_levels`,
`mts_level2_forces`, `mts_level2_factor`). The pre-2020 `mts_factor` key is
deprecated and silently ignored by grompp 2025 — space_v2 does not emit it.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import optuna  # noqa: F401
except Exception:  # pragma: no cover
    pass


def _budget_ps() -> float:
    """Trial MD length in picoseconds. Env-tunable so smoke stays short."""
    return float(os.environ.get("PIPELINE_MPS_BO_PS", "20000"))


# ---------------------------------------------------------------------------
# Categorical pools (12 v1 dims + 5 new v2 dims = 17 total)
# ---------------------------------------------------------------------------

# --- v2 NEW axes (previously frozen) --------------------------------------
DT_CHOICES = [0.001, 0.002, 0.003, 0.004, 0.005]      # ps; up from [0.001, 0.002]
HMR_CHOICES = [1.0, 2.5, 3.0, 3.5, 4.0]                # 1.0 = no repartition
LINCS_ORDER_CHOICES = [4, 6, 8]
LINCS_ITER_CHOICES = [1, 2]
MTS_FACTOR_CHOICES = [2, 3, 4]                          # mts-level2-factor

# --- v1 axes (unchanged so v2 winner is comparable to v1) ------------------
INTEGRATORS_NON_MTS = ["md", "md-vv", "sd"]
CONSTRAINTS = ["h-bonds", "h-angles", "all-bonds"]
PCOUPL_MDVV = ["C-rescale", "Berendsen"]                        # md-vv incompatible with PR
PCOUPL_MD_SD = ["Parrinello-Rahman", "C-rescale", "Berendsen"]
TCOUPL_WITH_PR = ["v-rescale", "nose-hoover"]                    # PR needs a real T
TCOUPL_WITHOUT_PR = ["v-rescale", "nose-hoover", "berendsen"]
NSTLIST_CHOICES = [10, 20, 40, 80]
NSTPCOUPLE_POOL = [10, 20]
NSTCALCENERGY_POOL = [50, 100]
RCUT_AMBER = [0.9, 1.0, 1.1]
FOURIER_CHOICES = [0.10, 0.12, 0.14, 0.16]
TAU_T_CHOICES = [0.1, 0.5, 1.0, 2.0]
TAU_P_CHOICES = [1.0, 2.0, 5.0, 10.0]
TAU_P_PR_CHOICES = [5.0, 10.0]


# ---------------------------------------------------------------------------
# Main suggest hook
# ---------------------------------------------------------------------------


def suggest_config(trial: "optuna.trial.Trial") -> dict[str, Any]:  # type: ignore[name-defined]
    """Emit a v2 MD production config for one Optuna trial.

    Hard grompp-fatal combos are avoided by predicate; empirical
    "shouldn't work at this dt / HMR" combos are NOT gated — grompp /
    mdrun errors are the ground truth and are captured downstream.
    """
    # === Tier 1: FF frozen (same as v1 — the seed topology is pre-equilibrated)
    protein_ff = "ff14SB"
    ligand_ff = "gaff2"
    water_model = "tip3p"

    # === Tier 2: cutoffs (independent)
    rcut = trial.suggest_categorical("rcut", RCUT_AMBER)
    fourierspacing = trial.suggest_categorical("fourierspacing", FOURIER_CHOICES)

    # === Tier 3: integrator (mts predicate below forces md)
    integrator = trial.suggest_categorical("integrator", INTEGRATORS_NON_MTS)

    # === Tier 4: MTS on/off — only valid with integrator=md AND constraints != none
    #     constraints=none is not in our pool so the second guard is a no-op,
    #     but we keep the predicate explicit for future extension.
    if integrator == "md":
        mts = trial.suggest_categorical("mts", [False, True])
    else:
        mts = False

    if mts:
        mts_level2_factor = trial.suggest_categorical("mts_level2_factor", MTS_FACTOR_CHOICES)
    else:
        mts_level2_factor = None

    # === Tier 5: neighbor-list stride (must divide mts factor when MTS is on)
    if mts:
        # nstlist must be a multiple of mts_level2_factor per grompp; filter pool.
        nstlist_pool = [x for x in NSTLIST_CHOICES if x % mts_level2_factor == 0]
        # Guaranteed non-empty for MTS_FACTOR_CHOICES = {2,3,4} and NSTLIST {10,20,40,80}
        nstlist = trial.suggest_categorical("nstlist", nstlist_pool)
    else:
        nstlist = trial.suggest_categorical("nstlist", NSTLIST_CHOICES)

    # === Tier 6: constraints + HMR (physically coupled to dt but we don't gate)
    constraints = trial.suggest_categorical("constraints", CONSTRAINTS)
    hmr_factor = trial.suggest_categorical("hmr_factor", HMR_CHOICES)

    # === Tier 7: LINCS — accuracy tunable at large dt
    lincs_order = trial.suggest_categorical("lincs_order", LINCS_ORDER_CHOICES)
    lincs_iter = trial.suggest_categorical("lincs_iter", LINCS_ITER_CHOICES)

    # === Tier 8: timestep — full envelope. Errors are the point.
    dt = trial.suggest_categorical("dt", DT_CHOICES)

    # === Tier 9: barostat (integrator-constrained), then thermostat
    if integrator == "md-vv":
        pcoupl = trial.suggest_categorical("pcoupl_mdvv", PCOUPL_MDVV)
    else:  # md or sd
        pcoupl = trial.suggest_categorical("pcoupl", PCOUPL_MD_SD)

    if integrator == "sd":
        tcoupl = "no"   # grompp forces this; not sampled to save budget
    elif pcoupl == "Parrinello-Rahman":
        tcoupl = trial.suggest_categorical("tcoupl_pr", TCOUPL_WITH_PR)
    else:
        tcoupl = trial.suggest_categorical("tcoupl", TCOUPL_WITHOUT_PR)

    # === Tier 10: coupling time constants (tau_p range tied to pcoupl)
    tau_t = trial.suggest_categorical("tau_t", TAU_T_CHOICES)
    if pcoupl == "Parrinello-Rahman":
        tau_p = trial.suggest_categorical("tau_p_pr", TAU_P_PR_CHOICES)
    else:
        tau_p = trial.suggest_categorical("tau_p", TAU_P_CHOICES)

    # === Tier 11: output cadence
    nstpcouple = trial.suggest_categorical("nstpcouple", NSTPCOUPLE_POOL)
    nstcalcenergy = trial.suggest_categorical("nstcalcenergy", NSTCALCENERGY_POOL)
    nstenergy = nstcalcenergy * 5

    # === Assemble MDP override -------------------------------------------------
    mdp: dict[str, Any] = {
        "integrator": integrator,
        "dt": dt,
        "nsteps": int(_budget_ps() / dt),
        "comm_mode": "Linear",
        "nstcomm": 100,
        "continuation": "yes",
        "gen_vel": "no",
        "constraints": constraints,
        "constraint_algorithm": "LINCS",
        "lincs_order": lincs_order,
        "lincs_iter": lincs_iter,
        "lincs_warnangle": 30.0,
        "cutoff_scheme": "Verlet",
        "nstlist": nstlist,
        "pbc": "xyz",
        "verlet_buffer_tolerance": 0.005,
        "rlist": rcut,
        "coulombtype": "PME",
        "rcoulomb": rcut,
        "fourierspacing": fourierspacing,
        "pme_order": 4,
        "ewald_rtol": 1e-5,
        "vdwtype": "Cut-off",
        "rvdw": rcut,
        "vdw_modifier": "Potential_Shift",
        "dispcorr": "EnerPres",
        "tcoupl": tcoupl,
        "tc_grps": "System",
        "ref_t": 300.0,
        "tau_t": tau_t,
        "nhchainlength": 10,
        "pcoupl": pcoupl,
        "pcoupltype": "isotropic",
        "ref_p": 1.0,
        "tau_p": tau_p,
        "nstpcouple": nstpcouple,
        "compressibility": 4.5e-5,
        "nstxout": 0,
        "nstvout": 0,
        "nstfout": 0,
        "nstxout_compressed": 1000,
        "compressed_x_precision": 1000,
        "nstenergy": nstenergy,
        "nstlog": 500,
        "nstcalcenergy": nstcalcenergy,
        "define": "",
        # HMR — runtime repartitioning (needs mdp.py schema field
        # mass_repartition_factor — already present).
        "mass_repartition_factor": hmr_factor,
        # MTS — only emit the full block when mts=True (mdp.py has the fields
        # as Optional[None]-defaulted so unemitted keys are correctly absent).
        "mts": mts,
    }
    if mts:
        mdp["mts_levels"] = 2
        mdp["mts_level2_forces"] = "longrange-nonbonded"
        mdp["mts_level2_factor"] = mts_level2_factor

    return {
        "mdp": mdp,
        "forcefield": {
            "protein_ff": protein_ff,
            "ligand_ff": ligand_ff,
            "water_model": water_model,
        },
    }
