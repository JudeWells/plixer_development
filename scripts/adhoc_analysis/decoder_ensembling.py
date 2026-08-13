"""Ensemble independently-seeded DECODERS on the likelihood-ranking metric.

`poc2mol_ensembling.py` averages across **Poc2Mol** checkpoints and scores the composition
(mass) readout. That axis is a no-op for the end-to-end frozen arms: every one of them shares
the same frozen `poc2mol_v2_11ch_ep576.ckpt`, so the upstreams are bitwise identical and only
the decoder differs. This script ensembles along the decoder axis instead, and scores the
thing the branch is judged on -- `val/likelihood_auc_znorm`.

Why this is worth running at all
--------------------------------
The end-to-end gradient was measured at +0.0002 (paired, 4 seeds, t = 0.02): a null. But the
same replication established that the frozen pipeline's run-to-run sigma is only 0.0047, so
the metric comfortably resolves effects of ~0.01 -- and CLAUDE.md §8/§20 records model
ensembling at **+0.030**, six times that noise floor. Eight independently-seeded frozen
decoders already exist as a by-product of the replication, so the members are free.

NORMALISATION -- the thing that makes the number honest
-------------------------------------------------------
Each member's (pocket x candidate) matrix is COLUMN z-normalised before averaging, which is
the same transform `likelihood_auc_znorm` applies. §20 records that row-standardising while
the metric z-normalises by column inflated every ensemble figure by ~+0.005 relative to its
baseline. The solo baselines printed here are computed from the SAME normalised matrices, so
the reported gain is a like-for-like comparison and not an artefact of the normalisation.

The per-member matrices are written to an .npz so `ensemble_subset_search.py` can be run post
hoc without repeating any forward passes. ⚠️ §20: greedy subset selection on the same panel
it is scored on is noise-fitting -- read the full-ensemble number, not the best subset.

Usage:
  ./venvPlixer/bin/python scripts/adhoc_analysis/decoder_ensembling.py \
      --run_dir logs/e2e_w12_frozen_anneal/runs/<ts> \
      --checkpoints a.ckpt,b.ckpt,... \
      --output ens_decoder.json --save_matrices ens_decoder.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import hydra                                                            # noqa: E402
from omegaconf import OmegaConf                                         # noqa: E402
from rdkit import Chem, RDLogger                                        # noqa: E402

from src.utils.likelihood_eval import (                                 # noqa: E402
    evaluate_likelihood_ranking,
    per_pocket_auc,
    znormalise_columns,
)

RDLogger.DisableLog("rdApp.*")


def decoy_panel(datamodule):
    """The shared candidate panel, in candidate order.

    Mirrors ``VoxToSmilesModel._candidate_smiles``, but reads the datamodule directly: that
    method goes through ``self.trainer``, which does not exist outside a Trainer.
    """
    for dataset in (getattr(datamodule, "val_datasets", {}) or {}).values():
        panel = getattr(dataset, "decoy_smiles_list", None)
        if panel:
            return list(panel)
    return None


def positive_mask_for(rows, smiles):
    """(P, N) bool marking each pocket's true ligand among the candidates.

    Matched by SMILES identity rather than row index, exactly as
    ``VoxToSmilesModel._log_likelihood_metrics`` does: duplicate SMILES in the panel would
    otherwise be scored as misses (CLAUDE.md 5.1).
    """
    positive = np.zeros((len(rows), len(rows[0][0])), dtype=bool)
    for row_ix, (_, binder_ix) in enumerate(rows):
        if smiles is not None and binder_ix < len(smiles):
            target = smiles[binder_ix]
            if target is not None:
                positive[row_ix] = np.array([s == target for s in smiles])
                continue
        positive[row_ix, binder_ix] = True
    return positive


def score_member(checkpoint, cfg, datamodule, loader, device):
    """Run one decoder over the panel and return its (pocket x candidate) score matrix.

    The scoring itself is delegated to the model's own ``_accumulate_likelihood_rows``, so
    this matrix is produced by literally the same code path that logs
    ``val/likelihood_auc_znorm`` during training. Reimplementing it would risk a number that
    is not comparable to the 0.7573 baseline it is being measured against.
    """
    model = hydra.utils.instantiate(cfg.model)
    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state.get("state_dict", state)
    # Non-strict: metric buffers (val_metrics.*) come and go between revisions. Anything
    # structural would show up as a missing decoder or poc2mol key, which is checked below.
    incompatible = model.load_state_dict(state_dict, strict=False)
    structural = [
        k for k in list(incompatible.missing_keys) + list(incompatible.unexpected_keys)
        if k.startswith(("model.", "poc2mol."))
    ]
    if structural:
        raise RuntimeError(
            f"{checkpoint}: state_dict does not match the config's architecture "
            f"({structural[:4]}). A channel-count mismatch is the usual cause."
        )

    model = model.eval().to(device)
    model._likelihood_rows = []
    # autocast is NOT optional here. Poc2Mol's master weights are fp32 (EndToEndPoc2Smiles
    # forces `.float()` so the optimiser update is not quantised away) while the voxel grids
    # arrive bf16 from the voxeliser, and the trainer reconciles the two with `bf16-mixed`.
    # Without it the first conv3d raises "Input type (c10::BFloat16) and bias type (float)
    # should be the same" -- and, worse, running the whole thing in fp32 instead would score
    # these checkpoints under different numerics from the ones that produced 0.7573.
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") \
        else torch.autocast("cpu", dtype=torch.bfloat16)
    with torch.no_grad(), autocast:
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch = datamodule.on_after_batch_transfer(batch)
            if "candidate_tokens" not in batch or "binder_indices" not in batch:
                continue
            pixel_values, _ = model.build_pixel_values(batch, training=False)
            model._accumulate_likelihood_rows(batch, pixel_values)

    rows = model._likelihood_rows
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dir", required=True,
                        help="a run directory containing resolved_config.yaml")
    parser.add_argument("--checkpoints", required=True, help="comma-separated .ckpt paths")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    parser.add_argument("--save_matrices", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run_dir, "resolved_config.yaml"))
    checkpoints = [c for c in args.checkpoints.split(",") if c]

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    loaders = datamodule.val_dataloader()
    # Dataloader 0 is roc_auc_plinder -- the only one carrying a decoy panel, and the one the
    # logged metric is computed from.
    loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders
    panel = decoy_panel(datamodule)

    matrices, solo = [], []
    positive = valid_columns = None
    for index, checkpoint in enumerate(checkpoints):
        rows = score_member(checkpoint, cfg, datamodule, loader, args.device)
        if not rows:
            raise RuntimeError(f"{checkpoint}: no likelihood rows -- is the decoy panel attached?")
        scores = np.stack([r for r, _ in rows])
        if positive is None:
            positive = positive_mask_for(rows, panel)
            if panel is not None:
                valid_columns = np.array([Chem.MolFromSmiles(s) is not None for s in panel])
        matrices.append(scores)
        metrics = evaluate_likelihood_ranking(scores, positive, valid_columns)
        solo.append(metrics["likelihood_auc_znorm"])
        print(f"[{index + 1}/{len(checkpoints)}] {os.path.basename(checkpoint)}  "
              f"solo znorm AUC {metrics['likelihood_auc_znorm']:.4f}  "
              f"(raw {metrics['likelihood_auc_raw']:.4f}, "
              f"blind {metrics['likelihood_auc_pocket_blind']:.4f}, "
              f"pockets {int(metrics['likelihood_n_pockets'])})", flush=True)

    # Column z-normalise EACH member, then average. Every member's columns then have unit sd,
    # so no member dominates by dynamic range, and the average is in the same space the
    # metric scores in.
    normalised = [znormalise_columns(m) for m in matrices]
    ensemble = np.mean(normalised, axis=0)
    ensemble_auc, n_pockets = per_pocket_auc(ensemble, positive, valid_columns)

    # The like-for-like solo baseline: same normalised matrices, scored one at a time.
    solo_normalised = [per_pocket_auc(m, positive, valid_columns)[0] for m in normalised]

    print()
    print(f"members                {len(matrices)}")
    print(f"pockets scored         {n_pockets}")
    print(f"solo znorm AUC mean    {np.mean(solo_normalised):.4f}  "
          f"(sd {np.std(solo_normalised, ddof=1):.4f}, "
          f"min {min(solo_normalised):.4f}, max {max(solo_normalised):.4f})")
    print(f"ENSEMBLE znorm AUC     {ensemble_auc:.4f}")
    print(f"gain over solo mean    {ensemble_auc - np.mean(solo_normalised):+.4f}")
    print(f"gain over best member  {ensemble_auc - max(solo_normalised):+.4f}")
    print()
    print("§20 reference: model ensembling measured at +0.030; frozen-pipeline seed sigma is")
    print("0.0047, so a gain below ~0.010 is not distinguishable from member-selection luck.")

    if args.save_matrices:
        np.savez_compressed(
            args.save_matrices,
            matrices=np.stack(matrices),
            positive=positive,
            valid_columns=valid_columns if valid_columns is not None else np.ones(matrices[0].shape[1], bool),
            checkpoints=np.array(checkpoints),
        )
        print(f"\nmember matrices -> {args.save_matrices}")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump({
                "checkpoints": checkpoints,
                "solo_znorm_auc": [float(s) for s in solo_normalised],
                "solo_znorm_auc_unnormalised_path": [float(s) for s in solo],
                "ensemble_znorm_auc": float(ensemble_auc),
                "n_pockets": int(n_pockets),
            }, handle, indent=2)
        print(f"summary -> {args.output}")


if __name__ == "__main__":
    main()
