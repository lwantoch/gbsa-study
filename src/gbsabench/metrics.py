#!/usr/bin/env python3
"""Metric + post-hoc library for the factorial GBSA analysis.

Everything downstream of the per-frame parquet lives here:
  * temporal slicing of per-frame ΔG  (length / stride / equil-offset)
  * entropy corrections IE and C2 computed from per-frame ΔGGAS (post-hoc)
  * ranking quality  : Kendall tau (predicted ΔG vs experimental pKi)
  * early enrichment : BEDROC (10 actives / 20 measured inactives — no decoys)
  * GBSA-vs-docking  : DeLong test (correlated ROC-AUC) + paired bootstrap (BEDROC)

Frame timing convention: the per-frame parquet is the 10-ps whole trajectory,
so frame index i (1-based) is at time i*10 ps. A cell (offset_ns, length_ns,
stride_ps) selects frames with  offset_ns <= t <= length_ns  every stride_ps.
"""
from __future__ import annotations
import numpy as np
from scipy import stats

KT = 0.001987204259 * 298.15   # kcal/mol at 298.15 K  (R in kcal/mol/K * T)
BASE_PS = 10                     # per-frame parquet spacing

# ---------------------------------------------------------------- temporal slice
def slice_frames(frame_idx: np.ndarray, offset_ns: float, length_ns: float, stride_ps: int):
    """Boolean mask over frame_idx (1-based, 10-ps spacing) for the temporal cell."""
    t_ns = frame_idx * BASE_PS / 1000.0
    keep = (t_ns > offset_ns) & (t_ns <= length_ns)
    step = max(1, stride_ps // BASE_PS)
    sel = np.zeros_like(frame_idx, dtype=bool)
    idx = np.where(keep)[0][::step]
    sel[idx] = True
    return sel

# ---------------------------------------------------------------- entropies
def ie_entropy(dgas: np.ndarray) -> float:
    """Interaction Entropy (Duan 2016): -TΔS = kT ln < exp(β (Egas-<Egas>)) >."""
    d = dgas - dgas.mean()
    # stable log-mean-exp
    x = d / KT
    m = x.max()
    return KT * (m + np.log(np.mean(np.exp(x - m))))

def c2_entropy(dgas: np.ndarray) -> float:
    """C2 entropy (Sun/Cournia 2018): -TΔS = σ²(Egas) / (2 kT)."""
    return np.var(dgas, ddof=1) / (2.0 * KT)

def predict_dg(dtotal: np.ndarray, dgas: np.ndarray, entropy: str) -> float:
    """Mean ΔG_bind for one ligand over the sliced frames, with optional -TΔS penalty."""
    dh = dtotal.mean()
    if entropy == "none": return dh
    if entropy == "IE":   return dh + ie_entropy(dgas)
    if entropy == "C2":   return dh + c2_entropy(dgas)
    raise ValueError(entropy)

# ---------------------------------------------------------------- ranking quality
def kendall_tau(pred_dg: np.ndarray, pchembl: np.ndarray) -> float:
    """τ between predicted affinity and experiment. Stronger binder = more negative
    ΔG = higher pKi, so correlate (-pred_dg) with pchembl."""
    if len(pred_dg) < 3: return np.nan
    return stats.kendalltau(-pred_dg, pchembl).statistic

# ---------------------------------------------------------------- BEDROC / AUC
# Use the validated RDKit + scikit-learn implementations rather than hand-rolled
# formulas (RDKit CalcBEDROC verified identical to the Truchon-Bayly closed form).
from rdkit.ML.Scoring.Scoring import CalcBEDROC as _CalcBEDROC, CalcRIE as _CalcRIE
from sklearn.metrics import roc_auc_score as _roc_auc_score

def bedroc(scores: np.ndarray, labels: np.ndarray, alpha: float = 20.0,
           higher_is_better: bool = True) -> float:
    """BEDROC via RDKit. scores = predicted BINDING STRENGTH (larger=better rank
    when higher_is_better); labels 1=active,0=decoy."""
    labels = np.asarray(labels)
    na = int(labels.sum())
    if na == 0 or na == len(labels): return np.nan
    order = np.argsort(-scores if higher_is_better else np.asarray(scores))
    sorted_lab = [[int(labels[i])] for i in order]      # RDKit: rows sorted best-first, col 0 = active flag
    return float(_CalcBEDROC(sorted_lab, 0, alpha))

def rie(scores: np.ndarray, labels: np.ndarray, alpha: float = 20.0,
        higher_is_better: bool = True) -> float:
    labels = np.asarray(labels)
    order = np.argsort(-scores if higher_is_better else np.asarray(scores))
    sorted_lab = [[int(labels[i])] for i in order]
    return float(_CalcRIE(sorted_lab, 0, alpha))

def auc_roc(scores: np.ndarray, labels: np.ndarray, higher_is_better: bool = True) -> float:
    labels = np.asarray(labels)
    if labels.sum() == 0 or labels.sum() == len(labels): return np.nan
    s = np.asarray(scores) if higher_is_better else -np.asarray(scores)
    return float(_roc_auc_score(labels, s))

# ---------------------------------------------------------------- DeLong test
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def delong_two_correlated(scores_a, scores_b, labels, higher_is_better=True):
    """DeLong test for two correlated ROC-AUCs on the SAME samples.
    Returns (auc_a, auc_b, z, p_two_sided). Sun & Xu (2014) fast implementation."""
    sa = scores_a if higher_is_better else -np.asarray(scores_a)
    sb = scores_b if higher_is_better else -np.asarray(scores_b)
    labels = np.asarray(labels)
    pos = labels == 1; neg = ~pos
    m = int(pos.sum()); n = int(neg.sum())
    if m == 0 or n == 0: return (np.nan, np.nan, np.nan, np.nan)
    # Full DeLong via placement values
    def compute(s):
        X = s[pos]; Y = s[neg]
        # placement values
        v01 = np.array([(np.sum(Y < x) + 0.5 * np.sum(Y == x)) / n for x in X])
        v10 = np.array([(np.sum(X > y) + 0.5 * np.sum(X == y)) / m for y in Y])
        auc = v01.mean()
        return auc, v01, v10
    aa, v01a, v10a = compute(sa)
    ab, v01b, v10b = compute(sb)
    s01 = np.cov(np.vstack([v01a, v01b]))
    s10 = np.cov(np.vstack([v10a, v10b]))
    S = s01 / m + s10 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return (aa, ab, np.nan, np.nan)
    z = (aa - ab) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return (float(aa), float(ab), float(z), float(p))

# ---------------------------------------------------------------- paired bootstrap BEDROC
def bootstrap_bedroc_diff(scores_a, scores_b, labels, alpha=20.0, n_boot=2000, seed_offset=0):
    """Paired bootstrap over ligands: distribution of BEDROC_a - BEDROC_b.
    Returns (diff_obs, ci_lo, ci_hi, p_two_sided). Deterministic per seed_offset."""
    labels = np.asarray(labels); N = len(labels)
    diff_obs = bedroc(scores_a, labels, alpha) - bedroc(scores_b, labels, alpha)
    rng = np.random.default_rng(12345 + seed_offset)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        la = labels[idx]
        if la.sum() == 0 or la.sum() == len(la):
            diffs[b] = np.nan; continue
        diffs[b] = bedroc(scores_a[idx], la, alpha) - bedroc(scores_b[idx], la, alpha)
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diff_obs), float(lo), float(hi), float(p)

# ---------------------------------------------------------------- selection-corrected max-T permutation
def _panel_bedroc_z(gbsa_scores_by_target, dock_scores_by_target, labels_by_target, alpha=20.0):
    """One-sided Wilcoxon z-statistic for the panel BEDROC test (GBSA > docking).

    gbsa/dock/labels are dicts keyed by target -> np.ndarray (ligand-aligned). A target
    contributes iff it has both actives and inactives after the given labelling. Returns
    the Wilcoxon signed-rank z (larger = stronger GBSA>docking evidence); NaN if <1 pair
    or all-zero differences (degenerate)."""
    gb, dk = [], []
    for tg in gbsa_scores_by_target:
        lab = labels_by_target[tg]
        na = int(lab.sum())
        if not (0 < na < len(lab)):
            continue
        bg = bedroc(gbsa_scores_by_target[tg], lab, alpha)
        bd = bedroc(dock_scores_by_target[tg], lab, alpha)
        if np.isnan(bg) or np.isnan(bd):
            continue
        gb.append(bg); dk.append(bd)
    gb, dk = np.asarray(gb), np.asarray(dk)
    if len(gb) < 1:
        return np.nan
    d = gb - dk
    if np.allclose(d, 0.0):
        return np.nan
    try:
        res = stats.wilcoxon(gb, dk, alternative="greater")
    except ValueError:
        return np.nan
    # map the one-sided p to a standard-normal z so larger statistic = more extreme
    return float(stats.norm.isf(res.pvalue))


def maxt_selection_corrected_p(gbsa_by_combo_target, dock_by_target, labels_by_target,
                               observed_best_combo, alpha=20.0, n_perm=5000, seed=20250827,
                               drop_targets=()):
    """Selection-corrected p for the best-of-K combos via a max-T label permutation.

    Under H0 (GBSA no better than docking at ranking actives) the active labels are
    exchangeable *within each target*. For each permutation we shuffle is_active within
    every target, recompute the panel BEDROC Wilcoxon z for ALL combos, and record the
    MAX z across combos. The selection-corrected p is the fraction of permutations whose
    max-z >= the observed z of the selected (best) combo. This controls the family-wise
    winner's-curse from having screened K correlated combos.

    Parameters
    ----------
    gbsa_by_combo_target : dict[combo][target] -> gbsa binding-strength array (ligand-aligned)
    dock_by_target       : dict[target]        -> docking binding-strength array (same order)
    labels_by_target     : dict[target]        -> is_active array (0/1), the OBSERVED labels
    observed_best_combo  : the combo selected as best on the observed data
    drop_targets         : iterable of target IDs to exclude (leave-one-target-out jackknife /
                           single-target sensitivity). observed_best_combo is held fixed at the
                           pre-specified combo, so the estimand is "how much does the corrected p
                           for the pre-specified combo rest on any one target".

    Returns (p_selection_corrected, observed_max_z, observed_selected_z, n_perm_effective).
    Deterministic given seed.
    """
    drop = set(drop_targets)
    if drop:
        gbsa_by_combo_target = {c: {t: v for t, v in d.items() if t not in drop}
                                for c, d in gbsa_by_combo_target.items()}
        dock_by_target = {t: v for t, v in dock_by_target.items() if t not in drop}
        labels_by_target = {t: v for t, v in labels_by_target.items() if t not in drop}
    combos = list(gbsa_by_combo_target.keys())
    targets = list(dock_by_target.keys())
    # observed statistics
    obs_z = {c: _panel_bedroc_z(gbsa_by_combo_target[c], dock_by_target, labels_by_target, alpha)
             for c in combos}
    obs_selected_z = obs_z[observed_best_combo]
    obs_max_z = np.nanmax(list(obs_z.values()))
    rng = np.random.default_rng(seed)
    ge = 0
    eff = 0
    for _ in range(n_perm):
        perm_labels = {tg: rng.permutation(labels_by_target[tg]) for tg in targets}
        zc = [_panel_bedroc_z(gbsa_by_combo_target[c], dock_by_target, perm_labels, alpha)
              for c in combos]
        zc = [z for z in zc if not np.isnan(z)]
        if not zc:
            continue
        eff += 1
        if np.max(zc) >= obs_selected_z:
            ge += 1
    # +1 correction (Phipson & Smyth 2010) to avoid p=0
    p = (ge + 1) / (eff + 1)
    return float(p), float(obs_max_z), float(obs_selected_z), int(eff)


def maxt_selection_corrected_p_signflip(gbsa_by_combo_target, dock_by_target,
                                        labels_by_target, observed_best_combo,
                                        alpha=20.0, n_perm=5000, seed=20250827,
                                        drop_targets=(), exact_max_targets=20):
    """Selection-corrected p for the best-of-K combos via a paired SIGN-FLIP max-T.

    Why this exists alongside ``maxt_selection_corrected_p``.  That function permutes
    ``is_active`` within each target, which destroys the active/inactive signal in the
    GBSA scores *and* in the docking scores at the same time.  The hypothesis it
    therefore tests is the joint one, "neither score carries information about
    activity".  That is not the hypothesis of interest here.

    The distinction bites in exactly the situation notebook 04 documents: on 5 of 8
    discovery targets Vina's whole-list AUC is below 0.5, i.e. docking is not merely
    uninformative but inverted.  Against a label-permutation null, a large
    ``BEDROC_gbsa - BEDROC_dock`` gap is "significant" even when GBSA itself is at
    chance, because the permutation also destroys docking's anti-predictiveness and so
    never generates the negative reference values that actually occur.

    The hypothesis we want is H0: *GBSA ranks actives no better than docking does*.
    Under that null the two scoring functions are exchangeable within a target, so the
    sign of each target's paired difference is exchangeable.  This routine builds the
    null by flipping those signs.  The same sign vector is applied across all K combos
    in a draw, which preserves the between-combo correlation that makes the max-T
    correction necessary in the first place.

    Returns (p_selection_corrected, observed_max_z, observed_selected_z, n_draws),
    matching the signature of the label-permutation version so the two can be reported
    side by side.  With T <= exact_max_targets the 2**T sign vectors are enumerated
    exactly, so the result is deterministic and seed-independent; above that it falls
    back to n_perm sampled draws.
    """
    drop = set(drop_targets)
    if drop:
        gbsa_by_combo_target = {c: {t: v for t, v in d.items() if t not in drop}
                                for c, d in gbsa_by_combo_target.items()}
        dock_by_target = {t: v for t, v in dock_by_target.items() if t not in drop}
        labels_by_target = {t: v for t, v in labels_by_target.items() if t not in drop}
    combos = list(gbsa_by_combo_target.keys())

    # Per-combo, per-target paired BEDROC differences under the OBSERVED labels.
    # Targets are aligned across combos so one sign vector can be shared by all of them.
    usable = [tg for tg in dock_by_target
              if 0 < int(np.asarray(labels_by_target[tg]).sum()) < len(labels_by_target[tg])]
    diffs = {}
    for c in combos:
        d = []
        for tg in usable:
            lab = labels_by_target[tg]
            bg = bedroc(gbsa_by_combo_target[c][tg], lab, alpha)
            bd = bedroc(dock_by_target[tg], lab, alpha)
            d.append(np.nan if (np.isnan(bg) or np.isnan(bd)) else bg - bd)
        diffs[c] = np.asarray(d, dtype=float)

    def _z(d):
        d = d[~np.isnan(d)]
        if len(d) < 1 or np.allclose(d, 0.0):
            return np.nan
        try:
            res = stats.wilcoxon(d, alternative="greater")
        except ValueError:
            return np.nan
        return float(stats.norm.isf(res.pvalue))

    obs_z = {c: _z(diffs[c]) for c in combos}
    obs_selected_z = obs_z[observed_best_combo]
    obs_max_z = np.nanmax(list(obs_z.values()))

    # With T targets there are only 2**T distinct sign vectors. At T = 8 that is 256, so
    # drawing 5000 of them resamples the same 256 points ~20x each and buys nothing but
    # Monte-Carlo noise -- the very saturation failure this package diagnoses elsewhere
    # (two referees, round 2). Below the threshold the null is ENUMERATED EXACTLY and the
    # p-value is a true permutation p with no seed dependence at all.
    T = len(usable)
    exact = T <= exact_max_targets
    if exact:
        sign_vectors = (1.0 - 2.0 * ((np.arange(2 ** T)[:, None] >> np.arange(T)) & 1))
    else:
        rng = np.random.default_rng(seed)
        sign_vectors = rng.choice((-1.0, 1.0), size=(n_perm, T))

    ge = eff = 0
    for signs in sign_vectors:
        zc = [_z(diffs[c] * signs) for c in combos]
        zc = [z for z in zc if not np.isnan(z)]
        if not zc:
            continue
        eff += 1
        if np.max(zc) >= obs_selected_z:
            ge += 1
    # Exact enumeration needs no +1 correction -- the observed sign vector is already in
    # the enumeration, so ge/eff is exact. Sampling keeps Phipson & Smyth (2010).
    p = ge / eff if exact else (ge + 1) / (eff + 1)
    return float(p), float(obs_max_z), float(obs_selected_z), int(eff)
