"""Optuna study driver.

Usage:
    python -m pipeline_mps.bayes_opt.optimize \
        --study my_first \
        --n-trials 100 \
        --mode throughput \
        --target 5HU9 \
        --ligand ligand01 \
        --storage sqlite:///$LUSTRE/pipeline-mps/bayes_opt.db

The SQLite storage lets the study resume, be inspected by ``optuna-dashboard``,
and (importantly) be shared across concurrent workers — you can launch this
same command from N SLURM tasks pointing at the same DB and they'll cooperate.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from pipeline_mps.bayes_opt.objective import FitnessMode, run_trial
from pipeline_mps.bayes_opt.space import suggest_config

log = logging.getLogger("pipeline_mps.bayes_opt")


def _make_objective(
    *,
    target: str,
    ligand: str,
    study_name: str,
    mode: FitnessMode,
    keep_workspace: bool,
):
    def _objective(trial):  # type: ignore[no-untyped-def]
        cfg = suggest_config(trial)
        # Attach the config verbatim so it shows up in optuna's UI / DB rows
        trial.set_user_attr("cfg", cfg)
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
    parser.add_argument("--study", required=True, help="Optuna study name")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--mode",
        choices=("md_production_wall", "throughput", "stability"),
        default="md_production_wall",
        help="Fitness function. Default: md_production_wall (1/hours of the "
             "md_production stage only). Legacy: throughput=ns/day, "
             "stability=1/(1+E-drift).",
    )
    parser.add_argument("--target", default="5HU9")
    parser.add_argument("--ligand", default="ligand01")
    parser.add_argument(
        "--storage",
        default="sqlite:////mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/pipeline-mps/bayes_opt.db",
        help="Optuna storage URL (sqlite path or postgres://...). Shared "
             "across concurrent workers.",
    )
    parser.add_argument("--sampler", choices=("tpe", "random"), default="tpe")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep-workspace", action="store_true",
                        help="Don't delete trajectory files after fitness extraction.")
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
            constant_liar=True,   # cooperate cleanly with concurrent workers
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
    objective = _make_objective(
        target=args.target,
        ligand=args.ligand,
        study_name=args.study,
        mode=args.mode,
        keep_workspace=args.keep_workspace,
    )
    study.optimize(objective, n_trials=args.n_trials, catch=(Exception,))

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        log.warning(
            "no COMPLETE trials in study %s (all %d attempts failed / pruned) — "
            "nothing to report", args.study, len(study.trials),
        )
        return 1
    log.info("best value = %.6f", study.best_value)
    log.info("best params = %s", study.best_params)
    log.info("best user_attr cfg = %s", study.best_trial.user_attrs.get("cfg"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
