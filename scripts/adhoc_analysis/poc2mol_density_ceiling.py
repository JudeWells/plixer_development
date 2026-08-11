"""How much ligand identity is recoverable from Poc2Mol's predicted density, without learning?

Motivation
----------
The stage-3 decoder ranks the true ligand well above decoys (z-norm AUC ~0.74) while barely
reconstructing anything (exact match 0.5%, Tanimoto 0.14). Those coexist if the predicted
density identifies a molecule's *gross properties* but not its structure. If so, the ceiling is
Poc2Mol's fidelity, not decoder overfitting, and no amount of regularisation moves it.

This measures the non-learned floor: score each candidate purely by how well its elemental
composition matches the predicted density's per-channel occupancy, then run the SAME per-pocket
AUC used for the decoder. Comparing the two says how much the decoder adds over "read the
formula off the density".

Why composition and not voxel overlap
------------------------------------
The candidate panel is SMILES only -- no 3D pose for decoys in this pocket's frame. Overlaying a
decoy's own grid would compare an arbitrary orientation against the true ligand's grid, which is
by construction the exact target Poc2Mol was trained to predict. That comparison is rigged and
would report an inflated ceiling. Per-channel occupancy mass is rotation- and translation-
invariant, needs no pose, and is available to the decoder too -- so it is the fair floor.

Two scores are reported:
  size+composition : cosine between the raw 9-vectors. Carries molecular size, which is a large
                     part of what the raw likelihood metric responds to.
  composition-only : cosine after normalising both to unit sum, so size is removed and only the
                     element *proportions* remain.

Usage:
    python scripts/adhoc_analysis/poc2mol_density_ceiling.py [--batch_size 32] [--output x.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import hydra  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

from src.utils.likelihood_eval import evaluate_likelihood_ranking  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Must mirror VoxelizationConfig.ligand_channels. Channel 8 is the catch-all for anything not
# listed (hydrogen is excluded from it, and appears in no channel at all).
CHANNEL_ELEMENTS = ["C", "O", "N", "S", "Cl", "F", "I", "Br"]
NAMED = set(CHANNEL_ELEMENTS) | {"H"}


def composition_vector(smiles):
    """Heavy-atom counts per ligand channel, or None if the SMILES does not parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    counts = np.zeros(len(CHANNEL_ELEMENTS) + 1)
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == "H":
            continue
        if symbol in CHANNEL_ELEMENTS:
            counts[CHANNEL_ELEMENTS.index(symbol)] += 1
        else:
            counts[-1] += 1  # catch-all
    return counts


def cosine_matrix(predicted, candidates):
    """(P,9) x (N,9) -> (P,N) cosine similarity: COMPOSITION only.

    Cosine is invariant to positive rescaling of either vector, so this carries element
    proportions and nothing about molecular size. (Normalising to unit sum first, as an
    earlier version did, is exactly such a rescale and therefore a no-op -- it produced
    bit-identical numbers to the unnormalised version.)
    """
    a = predicted / np.clip(np.linalg.norm(predicted, axis=1, keepdims=True), 1e-9, None)
    b = candidates / np.clip(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-9, None)
    return a @ b.T


def negative_distance_matrix(predicted, candidates, scale):
    """-L2 between calibrated predicted mass and candidate counts: size AND composition.

    `scale` converts occupancy mass to atom-count units per channel. It is fitted on this
    same set (mean mass / mean true count), so it is a mild in-sample advantage for this
    baseline -- noted rather than corrected, since the baseline is meant to be generous.
    """
    calibrated = predicted / np.clip(scale, 1e-9, None)[None, :]
    return -np.linalg.norm(calibrated[:, None, :] - candidates[None, :, :], axis=2)


def size_only_matrix(predicted_total, candidate_totals, scale):
    """-|predicted heavy-atom count - candidate heavy-atom count|. Size alone, no composition."""
    estimate = predicted_total / scale
    return -np.abs(estimate[:, None] - candidate_totals[None, :])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("PROJECT_ROOT", root)
    with hydra.initialize_config_dir(version_base="1.3",
                                     config_dir=os.path.join(root, "configs")):
        cfg = hydra.compose(config_name="train", overrides=[
            "experiment=exp1_s3_protein",
            "data.num_workers=4",
            f"data.config.batch_size={args.batch_size}",
            f"data.config.val_batch_size={args.batch_size}",
            "data.config.target_samples_per_batch=32",
            "paths.output_dir=/tmp/ceiling", "paths.img_save_dir=/tmp/ceiling/img",
        ])
    os.makedirs("/tmp/ceiling/img", exist_ok=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")

    # Dataloader 0 is roc_auc_plinder -- the one carrying the shared candidate panel, so the
    # AUC here is directly comparable with val/likelihood_auc_* from training.
    loader = datamodule.val_dataloader()[0]
    panel = list(datamodule.val_datasets["roc_auc_plinder"].decoy_smiles_list)

    predicted_mass, binder_indices = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
            # training=False forces fraction=1.0, i.e. Poc2Mol's PREDICTION rather than the
            # true ligand voxels. Going through datamodule.on_after_batch_transfer instead
            # would see trainer=None -> training=True, global_step=0 -> fraction 0, and hand
            # back ground-truth density: the exact opposite of what is being measured.
            out = datamodule.voxel_builder(batch, apply_quality_filter=False,
                                           training=False, global_step=0)
            ligand_density = out["pixel_values"][:, :9]          # predicted, [0,1]
            predicted_mass.append(ligand_density.sum(dim=(2, 3, 4)).float().cpu().numpy())
            binder_indices.extend(int(i) for i in out["binder_indices"])

    predicted_mass = np.concatenate(predicted_mass, axis=0)
    n_pockets = predicted_mass.shape[0]

    candidates = [composition_vector(s) for s in panel]
    valid_columns = np.array([c is not None for c in candidates])
    candidate_matrix = np.stack([c if c is not None else np.zeros(9) for c in candidates])

    positive = np.zeros((n_pockets, len(panel)), dtype=bool)
    for row, binder in enumerate(binder_indices):
        target = panel[binder] if binder < len(panel) else None
        positive[row] = np.array([s == target for s in panel]) if target else False

    results = {"n_pockets": n_pockets, "n_candidates": len(panel),
               "n_valid_candidates": int(valid_columns.sum())}

    # Fidelity: does the predicted per-channel mass track the TRUE ligand's atom counts?
    true_counts = np.stack([candidate_matrix[b] for b in binder_indices])

    # occupancy mass -> atom-count units, per channel
    scale = predicted_mass.mean(axis=0) / np.clip(true_counts.mean(axis=0), 1e-9, None)
    total_scale = predicted_mass.sum(1).mean() / max(true_counts.sum(1).mean(), 1e-9)

    scorers = {
        "size_only": size_only_matrix(predicted_mass.sum(1), candidate_matrix.sum(1), total_scale),
        "composition_only": cosine_matrix(predicted_mass, candidate_matrix),
        "size+composition": negative_distance_matrix(predicted_mass, candidate_matrix, scale),
    }
    for label, scores in scorers.items():
        metrics = evaluate_likelihood_ranking(scores, positive, valid_columns)
        results[label] = metrics
        print(f"\n--- {label} ---")
        for key, value in metrics.items():
            print(f"    {key:32s} {value}")

    np.savez("/tmp/ceiling_arrays.npz", predicted_mass=predicted_mass,
             candidate_matrix=candidate_matrix, positive=positive,
             valid_columns=valid_columns, true_counts=true_counts)

    print("\n--- density fidelity: predicted channel mass vs true atom count ---")
    per_channel = {}
    for c, name in enumerate(CHANNEL_ELEMENTS + ["other"]):
        if true_counts[:, c].std() < 1e-9:
            per_channel[name] = None
            print(f"    {name:6s} (constant in this set)")
            continue
        r = float(np.corrcoef(predicted_mass[:, c], true_counts[:, c])[0, 1])
        per_channel[name] = r
        print(f"    {name:6s} r = {r:+.3f}   mean predicted mass {predicted_mass[:, c].mean():8.2f}"
              f"   mean true count {true_counts[:, c].mean():6.2f}")
    total_r = float(np.corrcoef(predicted_mass.sum(1), true_counts.sum(1))[0, 1])
    results["fidelity_per_channel_r"] = per_channel
    results["fidelity_total_mass_r"] = total_r
    print(f"    {'TOTAL':6s} r = {total_r:+.3f}  (predicted total mass vs heavy-atom count)")

    print("\nReference — the trained decoder on this same panel: raw 0.5538, z-norm 0.7449.")
    print("If composition alone approaches that, the decoder is mostly reading the formula and")
    print("the ceiling is Poc2Mol's density fidelity, not decoder overfitting.")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
