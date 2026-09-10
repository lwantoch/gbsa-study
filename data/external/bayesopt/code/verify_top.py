"""Stage 2 of the two-stage BO: verify the top-N configs at 20 ns.

Stage 1 ran short (2 ns) trials to rank throughput. Stage 2 takes the top-N
completed+stable trials from Stage 1 and re-runs each at 20 ns to confirm the
config stays stable and its ns/hour holds up at production length.

Usage:
    python -m pipeline_mps.bayes_opt.verify_top \\
        --study md_prod_v1 --top-n 5 --target 5HU9 --ligand ligand01

Output: prints a summary table + writes verify_top_result.json in the study
root with 2ns fitness vs 20ns fitness, stability at both lengths.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import optuna

from pipeline_mps.bayes_opt.objective import STUDIES_ROOT, run_trial

log = logging.getLogger("verify_top")


def _load_top_configs(study_name: str, top_n: int) -> list[dict[str, Any]]:
    """Pull top-N COMPLETE trials from the study, sorted by fitness DESC.

    Only trials that also passed the stability filter (fitness > 0 is a good
    proxy — _fitness_md_production_wall returns 0.0 when stability fails).
    Reads cfg from each trial's user_attr where the objective stored it.
    """
    db = "sqlite:///" + str(Path.cwd() / "bayes_opt.db")
    study = optuna.load_study(study_name=study_name, storage=db)
    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value and t.value > 0
    ]
    complete.sort(key=lambda t: t.value, reverse=True)
    winners = complete[:top_n]

    if not winners:
        raise SystemExit(f"no COMPLETE+stable trials in study {study_name}")

    configs: list[dict[str, Any]] = []
    for t in winners:
        # cfg is stored in user_attrs by objective.py after suggest_config()
        cfg = t.user_attrs.get("cfg")
        if not cfg:
            # Fallback: reconstruct from trial_result.json in workspace
            trial_dirs = list(
                (STUDIES_ROOT / study_name).glob(
                    f"trial_{t.number:04d}/workspace/*/trial_result.json"
                )
            )
            if not trial_dirs:
                log.warning("no cfg for trial %d, skipping", t.number)
                continue
            with trial_dirs[0].open() as f:
                cfg = json.load(f)["cfg"]

        configs.append({
            "stage1_trial": t.number,
            "stage1_fitness": t.value,
            "cfg": cfg,
        })
    return configs


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, help="Stage 1 study name")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--target", default="5HU9")
    parser.add_argument("--ligand", default="ligand01")
    parser.add_argument("--verify-ps", type=int, default=20000,
                        help="MD budget for verify runs (ps, default 20000)")
    parser.add_argument(
        "--verify-study-suffix", default="_verify20ns",
        help="New study_name suffix for verify workspaces",
    )
    args = parser.parse_args()

    configs = _load_top_configs(args.study, args.top_n)
    log.info("verifying %d configs from %s at %d ps",
             len(configs), args.study, args.verify_ps)

    # Bump nsteps in each cfg to the verify budget.
    for c in configs:
        dt = c["cfg"]["mdp"]["dt"]
        c["cfg"]["mdp"]["nsteps"] = int(args.verify_ps / dt)
        log.info("  trial %d: dt=%.4f fs → nsteps=%d",
                 c["stage1_trial"], dt, c["cfg"]["mdp"]["nsteps"])

    verify_study = args.study + args.verify_study_suffix
    results: list[dict[str, Any]] = []

    for idx, c in enumerate(configs):
        log.info("[%d/%d] verify config from stage1 trial %d",
                 idx + 1, len(configs), c["stage1_trial"])
        try:
            fitness = run_trial(
                trial_number=idx,
                cfg=c["cfg"],
                target=args.target,
                ligand=args.ligand,
                study_name=verify_study,
                mode="md_production_wall",
                keep_workspace=True,
            )
        except Exception as e:
            log.exception("verify run failed for trial %d", c["stage1_trial"])
            fitness = 0.0
            failure = str(e)
        else:
            failure = None

        # Read the trial_result.json we just wrote to pull stability info.
        trial_dir = next(
            (STUDIES_ROOT / verify_study).glob(
                f"trial_{idx:04d}/workspace/*/trial_result.json"
            ),
            None,
        )
        stability = {}
        if trial_dir:
            with trial_dir.open() as f:
                tr = json.load(f)
            stability = {
                "stability_passed": tr.get("stability_passed"),
                "stability_reason": tr.get("stability_reason"),
                "md_production_wall_seconds": tr.get("md_production_wall_seconds"),
            }

        results.append({
            "verify_slot": idx,
            "stage1_trial": c["stage1_trial"],
            "stage1_fitness": c["stage1_fitness"],
            "verify_fitness": fitness,
            "verify_ps": args.verify_ps,
            **stability,
            "cfg": c["cfg"],
            "failure": failure,
        })

    out_path = STUDIES_ROOT / verify_study / "verify_top_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print()
    print("=" * 88)
    print(f"Stage 2 verify — {args.verify_ps} ps re-runs of top {len(results)}")
    print("=" * 88)
    print(f"{'slot':>4} {'s1_trial':>8} {'s1_fit':>10} {'v_fit':>10} "
          f"{'wall_s':>9} {'stable':>7}  reason")
    for r in results:
        stable = "yes" if r.get("stability_passed") else "NO"
        reason = (r.get("stability_reason") or "")[:40]
        wall = r.get("md_production_wall_seconds")
        print(f"{r['verify_slot']:>4} {r['stage1_trial']:>8} "
              f"{r['stage1_fitness']:>10.4g} {r['verify_fitness']:>10.4g} "
              f"{wall or 0:>9.0f} {stable:>7}  {reason}")
    print(f"\nfull results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
