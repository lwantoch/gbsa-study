"""Bayesian optimization over MD production parameters.

Search-space + trial-runner + optuna study loop that respects the physical
constraints between MD knobs (e.g. HMR-only dt=4fs, MTS requires leap-frog,
Parrinello-Rahman only outside equilibration).

Entry points:
- ``pipeline_mps.bayes_opt.space.suggest_config(trial)`` — Optuna hook that
  emits a valid MDP + FF override dict for one trial.
- ``pipeline_mps.bayes_opt.objective.run_trial(cfg, ...)`` — turns a config
  into a signac job, runs the pipeline, returns the fitness scalar.
- ``python -m pipeline_mps.bayes_opt.optimize --study foo --n-trials 100``
  — the study driver.
"""
