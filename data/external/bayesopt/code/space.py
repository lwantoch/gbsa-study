"""Constraint-by-construction Optuna search space for MD production.

Rewritten 2026-08-31 after deep GROMACS-grompp compatibility research: every
``trial.suggest_*`` call is guarded by predicates on already-fixed choices, so
the sampled config is physically valid by construction. **No ``TrialPruned``
paths** — TPE never wastes an evaluation on an impossible corner.

The 18 sampled knobs (grouped by their dependency tier):

  Tier 1  Force-field family        protein_ff
          Water model               water_model         (constrained by protein_ff)
          Ligand FF                 ligand_ff           (constrained by protein_ff)

  Tier 2  Nonbonded cutoff          rcut                (range depends on FF family)
          PME grid                  fourierspacing

  Tier 3  MTS toggle                mts
          MTS factor                mts_factor          (only if mts)
          Integrator                integrator          (mts forces md)
          Neighbor list             nstlist             (must divide mts_factor)

  Tier 4  Constraints               constraints         (h-bonds+ only; none dropped)
          HMR factor                hmr_factor

  Tier 5  Timestep                  dt                  (range depends on hmr_factor)

  Tier 6  Barostat                  pcoupl              (md-vv forbids Parrinello-R)
          Thermostat                tcoupl              (sd forces 'no'; PR forbids 'no')

  Tier 7  Coupling times            tau_t               (unconstrained)
                                    tau_p               (>=5 ps if Parrinello-Rahman)

  Tier 8  MTS output cadence        nstpcouple          (must divide mts_factor)
                                    nstcalcenergy       (must divide mts_factor)

Derived (not sampled):
  vdw_modifier   = 'Force-switch' if charmm36m else 'Potential-shift'
  rvdw_switch    = rcut - 0.2       (only if Force-switch)
  comm_mode      = 'Linear'         (Angular is grompp-warned in periodic solvent)

Hard-forbidden combos eliminated by construction (grompp-fatal per GROMACS 2025
docs + readir.cpp):

  - integrator=md-vv + pcoupl=Parrinello-Rahman
  - mts=True + integrator!=md
  - mts=True + constraints=none
  - mts=True + nstlist / nstpcouple / nstcalcenergy not divisible by mts_factor
  - pcoupl=Parrinello-Rahman + tcoupl=no
  - vdw_modifier=Force-switch + rvdw_switch >= rvdw
  - constraints=none + dt >= 2 fs
  - HMR > 1.0 + constraints=none

Physics-wrong combos also eliminated (grompp-silent but scientifically wrong):

  - ff19SB with any water != opc                       (Tian et al. JCTC 2020)
  - charmm36m with any water != CHARMM-tip3p           (Huang et al. NatMeth 2017)
  - charmm36m with vdw_modifier != Force-switch        (CHARMM parameterization)
  - AMBER FF with vdw_modifier != Potential-shift      (AMBER parameterization)
  - CGenFF with AMBER protein or GAFF2/OpenFF with CHARMM  (combining-rule mismatch)
  - dt >= 4 fs without HMR>=3.0                        (Hopkins et al. JCTC 2015)
  - dt >= 3 fs without HMR>=2.5

Silently-overridden knobs are not sampled:
  - tcoupl when integrator=sd (grompp forces 'no')
"""

from __future__ import annotations

import os
from typing import Any


def _budget_ps() -> float:
    """Trial MD length in picoseconds. Env-tunable so smoke runs stay short
    (e.g., PIPELINE_MPS_BO_PS=500 → 500 ps ≈ 2-3 min on A100) while the full
    study can use a longer window (default 20 ns) for a stable ns/hour signal."""
    return float(os.environ.get("PIPELINE_MPS_BO_PS", "20000"))

# Optuna is a soft dep — imported lazily so `import pipeline_mps.bayes_opt.space`
# doesn't require it. Trial protocol only.
try:
    import optuna  # noqa: F401
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# Discrete option pools (kept categorical for TPE efficiency on small budgets)
# ---------------------------------------------------------------------------

PROTEIN_FFS = ["ff14SB", "ff19SB"]

# Waters valid for ff14SB. ff19SB → opc only (Tian et al. 2020).
AMBER_WATERS = ["tip3p", "tip4p-ew", "spce"]

# Ligand FFs supported by pipeline_mps.parametrization today (see
# parametrization_enum.LigandFF). OpenFF/CGenFF are NOT in the search space
# because the parametrize layer can't consume them yet; when it can, add them
# here and the space.py picks them up automatically.
AMBER_LIGAND_FFS = ["gaff2"]

INTEGRATORS_NON_MTS = ["md", "md-vv", "sd"]

# constraints=none is dropped: never scientifically valid for dt >= 1 fs, and
# every knob-combo we care about (HMR, MTS, dt=2fs) requires at least h-bonds.
CONSTRAINTS = ["h-bonds", "h-angles", "all-bonds"]

# Barostat pools split by integrator compatibility.
PCOUPL_MDVV = ["C-rescale", "Berendsen"]                        # PR fatal with md-vv
PCOUPL_MD_SD = ["Parrinello-Rahman", "C-rescale", "Berendsen"]  # PR ok with md, sd

# Thermostat pools split by barostat. PR+no is fatal; berendsen+PR is warned but
# gives a wrong ensemble — dropped. sd bakes it in (see main function).
TCOUPL_WITH_PR = ["v-rescale", "nose-hoover"]                    # PR needs a real T
TCOUPL_WITHOUT_PR = ["v-rescale", "nose-hoover", "berendsen"]

# HMR and MTS were dropped 2026-08-31 after grompp errors on ff19SB topology:
# - Runtime HMR ("mass-repartition-factor") fails with "Light atoms are bound
#   to at least one atom that has a too low mass for repartitioning" because
#   the seed topology was built with default 1 amu H's and certain
#   ff19SB/OPC atom groups don't have enough neighbor mass to redistribute
#   from at run-time. Would need to rebuild the topology with pre-baked H
#   masses (setup_system change). Out of scope for this BO campaign.
# - MTS ("mts=yes / mts-factor") is currently unsupported by grompp: it
#   warns "Unknown left-hand 'mts-factor'" and silently ignores the setting.
#   The correct mdp key on GROMACS 2025 is `mts-factors` (plural); wiring
#   that through GromacsParams is a follow-up.
# Both branches are removed from the search space so no trial ever proposes
# an unusable knob. When rebuilding the topology or wiring `mts-factors`,
# extend HMR_FACTORS / add MTS_FACTORS back and the tier code re-activates.

# dt within h-bonds (2 fs) safety envelope without HMR.
DT_NO_HMR = [0.001, 0.002]

# nstpcouple / nstcalcenergy pools. C-rescale requires
# tau_p >= 25 * nstpcouple * dt; at dt=0.002, nstpcouple=10 satisfies every
# tau_p in {1, 2, 5, 10}. Keep the pool tight so warnings don't fire.
NSTPCOUPLE_POOL = [10, 20]
NSTCALCENERGY_POOL = [50, 100]

# Cutoff pool (AMBER parameterized at ~1.0 nm w/ Potential-shift).
RCUT_AMBER = [0.9, 1.0, 1.1]

FOURIER_CHOICES = [0.10, 0.12, 0.14, 0.16]

TAU_T_CHOICES = [0.1, 0.5, 1.0, 2.0]
TAU_P_CHOICES = [1.0, 2.0, 5.0, 10.0]
TAU_P_PR_CHOICES = [5.0, 10.0]  # Parrinello-Rahman needs slow coupling


# ---------------------------------------------------------------------------
# Main suggest hook
# ---------------------------------------------------------------------------


def suggest_config(trial: "optuna.trial.Trial") -> dict[str, Any]:  # type: ignore[name-defined]
    """Emit a physically valid MD production config for one Optuna trial.

    Returns a nested dict::

        {
          "mdp": {...},              # PRODUCTION_PARAMS override for md.py
          "forcefield": {
              "protein_ff": str,
              "ligand_ff":  str,
              "water_model": str,
          },
        }

    Every combination in the emitted config is grompp-valid AND physically
    consistent with the FF-family choice. No trial ever gets pruned for a
    physics-invalid combo.
    """
    # === Tier 1: FF choice FIXED to whatever the BO seed was equilibrated with.
    # BO trials re-run only md_production against a cached topology, so the
    # protein_ff/water_model/ligand_ff sp keys can't retroactively swap the FF
    # — including them in the search space would just waste TPE budget on
    # duplicate physics. Vary FF only when the seed is regenerated per trial
    # (out of scope for this campaign).
    protein_ff = "ff14SB"
    ligand_ff = "gaff2"
    water_model = "tip3p"
    vdw_modifier = "Potential_Shift"

    # === Tier 2: cutoffs (AMBER-tuned range; PME grid independent)
    rcut = trial.suggest_categorical("rcut", RCUT_AMBER)
    rvdw_switch = None
    fourierspacing = trial.suggest_categorical("fourierspacing", FOURIER_CHOICES)

    # === Tier 3: integrator + nstlist  (MTS branch removed; see module docstring)
    mts = False
    integrator = trial.suggest_categorical("integrator", INTEGRATORS_NON_MTS)
    nstlist = trial.suggest_categorical("nstlist", [10, 20, 40, 80])

    # === Tier 4: constraints (h-bonds+; none dropped). HMR disabled → factor 1.0.
    constraints = trial.suggest_categorical("constraints", CONSTRAINTS)
    hmr_factor = 1.0

    # === Tier 5: dt within h-bonds safety envelope (no HMR).
    dt = trial.suggest_categorical("dt", DT_NO_HMR)

    # === Tier 6: barostat (integrator-constrained) then thermostat
    if integrator == "md-vv":
        pcoupl = trial.suggest_categorical("pcoupl_mdvv", PCOUPL_MDVV)
    else:  # md or sd
        pcoupl = trial.suggest_categorical("pcoupl", PCOUPL_MD_SD)

    if integrator == "sd":
        # grompp silently overrides tcoupl to 'no' — don't waste a BO dim.
        tcoupl = "no"
    elif pcoupl == "Parrinello-Rahman":
        tcoupl = trial.suggest_categorical("tcoupl_pr", TCOUPL_WITH_PR)
    else:
        tcoupl = trial.suggest_categorical("tcoupl", TCOUPL_WITHOUT_PR)

    # === Tier 7: coupling time constants (tau_p range tied to pcoupl)
    tau_t = trial.suggest_categorical("tau_t", TAU_T_CHOICES)
    if pcoupl == "Parrinello-Rahman":
        tau_p = trial.suggest_categorical("tau_p_pr", TAU_P_PR_CHOICES)
    else:
        tau_p = trial.suggest_categorical("tau_p", TAU_P_CHOICES)

    # === Tier 8: output cadence — tight so C-rescale doesn't fire tau_p warning.
    nstpcouple = trial.suggest_categorical("nstpcouple", NSTPCOUPLE_POOL)
    nstcalcenergy = trial.suggest_categorical("nstcalcenergy", NSTCALCENERGY_POOL)

    # nstenergy must be a multiple of nstcalcenergy (grompp-enforced regardless
    # of MTS). Fix at 5x — matches the historic pipeline default nstenergy=500.
    nstenergy = nstcalcenergy * 5

    # === Assemble the MDP override dict
    #
    # `mts` and `mass_repartition_factor` are OMITTED (not emitted) — the
    # pipeline's default MDP already sets them to safe values and this BO
    # study is not exploring those axes yet. Setting them to explicit defaults
    # here would fire the "Unknown left-hand 'mts-factor'" grompp warning
    # and the mass-repartitioning ERROR.
    mdp: dict[str, Any] = {
        "integrator": integrator,
        "dt": dt,
        # nsteps = budget_ps / dt (BO fitness = 1 / md_production_wall). Budget
        # is env-tunable via PIPELINE_MPS_BO_PS (default 20 ns).
        "nsteps": int(_budget_ps() / dt),
        "comm_mode": "Linear",         # hard: Angular is grompp-warned with pbc=xyz
        "nstcomm": 100,
        "continuation": "yes",
        "gen_vel": "no",
        "constraints": constraints,
        "constraint_algorithm": "LINCS",
        "lincs_order": 4,
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
        "vdw_modifier": vdw_modifier,
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
    }
    if rvdw_switch is not None:
        mdp["rvdw_switch"] = rvdw_switch

    return {
        "mdp": mdp,
        "forcefield": {
            "protein_ff": protein_ff,
            "ligand_ff": ligand_ff,
            "water_model": water_model,
        },
    }
