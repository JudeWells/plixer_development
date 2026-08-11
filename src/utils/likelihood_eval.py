"""Pocket-conditioned likelihood ranking, raw and z-normalised.

The protocol: for each pocket, score the true ligand and a shared panel of decoys under
teacher forcing, then ask how highly the true ligand ranks. That gives a virtual-screening
style AUC per pocket.

Why the z-normalisation is not optional
---------------------------------------
The raw metric is dominated by a nuisance term. Decomposing the 943x943 pocket x ligand
matrix from the published run (CLAUDE.md 5.1):

    pocket effect     6.4%
    ligand effect    84.0%   <- intrinsic molecule likelihood, mostly SIZE
    interaction       9.5%   <- the only part that can encode pocket specificity

Likelihood here is mean per-token cross-entropy, so it tracks molecule size
(corr with heavy-atom count = -0.758). That term is pure noise for a *ranking* task -- a
pocket-blind baseline that scores by ligand mean alone gets AUC exactly 0.500 -- but it
swamps the interaction term that actually measures whether the model understands the pocket.

Subtracting each ligand's mean across pockets (z-scoring the columns) removes it. On the
published model this moved chrono 0.67 -> 0.854 and PLINDER 0.58 -> 0.753, and the controls
all passed: the column view gives 0.8493 independently, size-matched decoys still 0.804,
permutation null 0.543.

So: **quote the z-normalised number.** The raw one is reported alongside only for continuity
with the paper.

Requires a shared candidate panel across pockets, which is what
``Poc2MolOutputDataset(include_decoys=True)`` builds -- every pocket scores the same list, so
the scores form a dense matrix whose columns are comparable.
"""
from __future__ import annotations

import numpy as np


def znormalise_columns(scores):
    """Z-score each ligand (column) across pockets (rows).

    Removes the per-ligand offset -- the intrinsic "this molecule is likely under any
    pocket" term -- leaving only how a pocket moves that ligand relative to its own
    average. Columns with zero variance (or too few pockets) are returned centred but
    unscaled, which keeps them from becoming inf.

    Note the column statistics include the very entry being normalised. A leave-one-out
    variant is NOT worth implementing: substituting the closed-form LOO moments, the column
    mean and sd cancel and z_loo = u*sqrt(P/((P-1)(1-u^2/(P-1)))) with u the in-sample z --
    a strictly increasing function of u alone. Ranking within a pocket is therefore
    identical, so AUROC, EF and BEDROC are unchanged (verified: within-row Spearman exactly
    1.0, AUCs equal to all digits). Only non-rank uses of the z-values -- absolute
    thresholds, calibration, comparing confidence across pockets -- would see a difference.
    Normalising against an INDEPENDENT background panel of pockets is a genuinely different
    computation and can reorder a row; that is the open prospective-deployment question.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape[0] < 2:
        return scores - scores.mean(axis=0, keepdims=True)
    mean = scores.mean(axis=0, keepdims=True)
    std = scores.std(axis=0, ddof=0, keepdims=True)
    safe = np.where(std > 1e-12, std, 1.0)
    return (scores - mean) / safe


def _auc(pos_scores, neg_scores):
    """Mann-Whitney AUC with explicit tie handling (0.5 credit), no sklearn dependency."""
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return None
    pos = np.asarray(pos_scores)[:, None]
    neg = np.asarray(neg_scores)[None, :]
    wins = (pos > neg).sum() + 0.5 * (pos == neg).sum()
    return float(wins / (pos.size * neg.size))


def per_pocket_auc(scores, positive_mask, valid_columns=None):
    """Mean of the per-pocket AUCs.

    This is the convention the paper used (`calculate_autodock_vina_roc_auc.py` does the
    same), so it stays comparable. Per-system *medians* run much higher -- 0.739 vs 0.669
    on chrono -- so do not mix the two.

    Args:
        scores: (P, N) likelihood per pocket per candidate; higher = more likely.
        positive_mask: (P, N) bool, True where the candidate IS that pocket's true ligand.
            Matched by SMILES identity rather than row index on purpose: 27 test systems
            share a SMILES with another, and index matching would score those as misses.
        valid_columns: optional (N,) bool of candidates to score at all. Used to drop
            candidates whose SMILES does not parse -- the shipped PLINDER decoy panel has
            one corrupt entry.

    Returns:
        (mean_auc, n_pockets_scored). mean_auc is nan when nothing could be scored.
    """
    scores = np.asarray(scores, dtype=np.float64)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    if valid_columns is None:
        valid_columns = np.ones(scores.shape[1], dtype=bool)
    valid_columns = np.asarray(valid_columns, dtype=bool)

    aucs = []
    for row_scores, row_positive in zip(scores, positive_mask):
        pos = row_scores[row_positive & valid_columns]
        neg = row_scores[(~row_positive) & valid_columns]
        auc = _auc(pos, neg)
        if auc is not None:
            aucs.append(auc)
    if not aucs:
        return float("nan"), 0
    return float(np.mean(aucs)), len(aucs)


def evaluate_likelihood_ranking(scores, positive_mask, valid_columns=None):
    """Both views of the same matrix: raw, and with the ligand-size nuisance removed.

    Returns a dict of scalars ready to log. ``pocket_blind`` is the sanity control -- it
    scores every pocket by the ligand column mean alone, ignoring the pocket entirely, and
    must come out at ~0.5. If it does not, the matrix is malformed.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0:
        return {}

    raw_auc, n = per_pocket_auc(scores, positive_mask, valid_columns)
    znorm_auc, _ = per_pocket_auc(
        znormalise_columns(scores), positive_mask, valid_columns
    )

    # Pocket-blind control: replace every row by the column means. Any pocket-specific
    # signal is destroyed, so a correct implementation lands on 0.5.
    blind = np.tile(scores.mean(axis=0, keepdims=True), (scores.shape[0], 1))
    blind_auc, _ = per_pocket_auc(blind, positive_mask, valid_columns)

    return {
        "likelihood_auc_raw": raw_auc,
        "likelihood_auc_znorm": znorm_auc,
        "likelihood_auc_pocket_blind": blind_auc,
        "likelihood_n_pockets": float(n),
    }
