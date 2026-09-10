"""BO v2 study driver — uses `space_v2` and warm-starts from v1 top-N.

The v1 driver (`optimize.py`) is unchanged; this module is a sibling so v1
reproducibility isn't at risk of drift.

Usage:
    python -m pipeline_mps.bayes_opt.optimize_v2 \
        --study md_prod_v2 \
        --n-trials 200 \
        --warm-start-from md_prod_v1 \
        --warm-start-top 5

Each warm-start config from the v1 study is enqueued into v2 with
`hmr_factor = 1.0`, `lincs_order = 4`, `lincs_iter = 1`, `mts = False`
(the v2 defaults that reproduce v1 behavior) so TPE starts with a known
plateau of good configs and can then explore the new HMR/MTS/LINCS axes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

from pipeline_mps.bayes_opt.objective import FitnessMode, run_trial
from pipeline_mps.bayes_opt.space_v2 import (
    HMR_CHOICES,
    LINCS_ITER_CHOICES,
    LINCS_ORDER_CHOICES,
    MTS_FACTOR_CHOICES,
    suggest_config,
)

log = logging.getLogger("pipeline_mps.bayes_opt.v2")


def _load_v1_top_configs(db_path: Path, v1_study: str, top_n: int) -> list[dict[str, Any]]:
    """Read the top-N v1 trials directly from SQLite (avoids optuna load overhead)."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        """
        SELECT t.number, tv.value, tp.param_name, tp.param_value, tp.distribution_json
        FROM trials t
        JOIN trial_values tv ON tv.trial_id = t.trial_id
        JOIN studies s ON s.study_id = t.study_id
        JOIN trial_params tp ON tp.trial_id = t.trial_id
        WHERE s.study_name = ? AND t.state = 'COMPLETE' AND tv.value > 0
        ORDER BY tv.value DESC, t.number
        """,
        (v1_study,),
    ).fetchall()
    con.close()

    # Group by trial number preserving descending-fitness ordering
    seen: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for number, value, param_name, param_value, dist_json in rows:
        if number not in seen:
            seen[number] = {"__number__": number, "__fitness__": value}
            order.append(number)
        choices = json.loads(dist_json)["attributes"]["choices"]
        seen[number][param_name] = choices[int(param_value)]

    top = [seen[n] for n in order[:top_n]]
    log.info("loaded %d v1 top configs (fitness range %.3f-%.3f)",
             len(top),
             top[-1]["__fitness__"] if top else 0,
             top[0]["__fitness__"] if top else 0)
    return top


def _v1_config_to_v2_params(v1_cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert a v1 parameter dict to the v2 categorical-key space.

    v2 has additional axes (hmr_factor, lincs_order, lincs_iter, mts,
    mts_level2_factor). We seed them with the v2-defaults that reproduce
    v1 behavior: no HMR, LINCS order=4 iter=1, no MTS. TPE then explores
    the extended axes around this proven baseline.
    """
    # Strip bookkeeping fields
    v2 = {k: v for k, v in v1_cfg.items() if not k.startswith("__")}
    # Seed the new axes at values that map to "same as v1"
    v2.setdefault("hmr_factor", 1.0)
    v2.setdefault("lincs_order", 4)
    v2.setdefault("lincs_iter", 1)
    v2.setdefault("mts", False)
    # mts_level2_factor is only sampled when mts=True; leave absent for warm-start
    return v2


def _make_objective(
    *, target: str, ligand: str, study_name: str, mode: FitnessMode,
    keep_workspace: bool,
):
    def _objective(trial):  # type: ignore[no-untyped-def]
        cfg = suggest_config(trial)
        trial.set_user_attr("cfg", cfg)
        # Record the actually-attempted knobs for post-hoc failure analysis,
        # even for trials that error out — required by the "report errors,
        # don't skip" project rule.
        trial.set_user_attr("space_version", "v2")
        fitness = run_trial(
            trial_number=trial.number,
            cfg=cfg,
            target=target,
            ligand=ligand,
            study_name=study_name,
            mode=mode,
            keep_workspace=keep_workspace,
        )
        return fitness
    return _objective


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", required=True, help="v2 study name (e.g. md_prod_v2)")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument(
        "--mode",
        choices=("md_production_wall", "throughput", "stability"),
        default="md_production_wall",
    )
    parser.add_argument("--target", default="5HU9")
    parser.add_argument("--ligand", default="ligand01")
    parser.add_argument(
        "--storage",
        default="sqlite:////mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/pipeline-mps/bayes_opt.db",
    )
    parser.add_argument("--sampler", choices=("tpe", "random"), default="tpe")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--warm-start-from", default=None,
                        help="v1 study name to seed top-N configs from")
    parser.add_argument("--warm-start-top", type=int, default=5)
    args = parser.parse_args(argv)

    import optuna
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    optuna.logging.set_verbosity(optuna.logging.INFO)

    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=10,
            n_ei_candidates=32,
            seed=args.seed,
            constant_liar=True,
        )
    else:
        sampler = optuna.samplers.RandomSampler(seed=args.seed)

    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    # Record v2 space provenance on the study
    study.set_user_attr("space_module", "pipeline_mps.bayes_opt.space_v2")
    study.set_user_attr("hmr_choices", HMR_CHOICES)
    study.set_user_attr("lincs_order_choices", LINCS_ORDER_CHOICES)
    study.set_user_attr("lincs_iter_choices", LINCS_ITER_CHOICES)
    study.set_user_attr("mts_factor_choices", MTS_FACTOR_CHOICES)

    # Warm-start: enqueue the v1 top-N configs so TPE has a proven baseline
    if args.warm_start_from:
        db_path = Path(args.storage.replace("sqlite:///", ""))
        top = _load_v1_top_configs(db_path, args.warm_start_from, args.warm_start_top)
        for v1_cfg in top:
            v2_params = _v1_config_to_v2_params(v1_cfg)
            study.enqueue_trial(v2_params, user_attrs={
                "warm_start_from": args.warm_start_from,
                "v1_trial_number": v1_cfg["__number__"],
                "v1_fitness": v1_cfg["__fitness__"],
            }, skip_if_exists=True)
        log.info("enqueued %d warm-start trials from %s", len(top), args.warm_start_from)

    objective = _make_objective(
        target=args.target,
        ligand=args.ligand,
        study_name=args.study,
        mode=args.mode,
        keep_workspace=args.keep_workspace,
    )
    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    stable = [t for t in completed if (t.value or 0) > 0]
    log.info("v2 study %s: %d/%d complete, %d stable",
             args.study, len(completed), len(study.trials), len(stable))
    if stable:
        log.info("best value = %.6f", max(t.value for t in stable))
        best = max(stable, key=lambda t: t.value)
        log.info("best user_attr cfg = %s", best.user_attrs.get("cfg"))
    return 0 if stable else 1


if __name__ == "__main__":
    sys.exit(main())
