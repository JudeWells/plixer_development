"""Does a Poc2Mol checkpoint's predicted density discriminate the true ligand from decoys?

Scheme-agnostic. The ligand channel map is read from the checkpoint's OWN resolved config, so
a 9-channel and an 11-channel model are each scored under their own semantics -- which is the
only way to compare them, since their `val/loss` values are not comparable at all (the Dice
floor moves with how many channels a typical ligand occupies; CLAUDE.md §3c).

Two readouts, and the distinction between them matters:

  A. COMPOSITION (pose-free, PER-SCHEME). Score each candidate by how well its per-channel
     heavy-atom counts match the predicted density's per-channel occupancy mass, calibrated
     to atom-count units. This is §12a's readout generalised past the hardcoded 8-element map.

  B. SIZE ONLY (scheme-INDEPENDENT). Collapse every ligand channel to one total occupancy and
     score on estimated-vs-candidate heavy-atom count alone. Same units in both schemes.

  ⚠️ A is NOT directly comparable across schemes: 11 channels give the readout 11 descriptors
  to match on against 9, so a richer scheme can win on descriptor count alone rather than on
  density quality. B is the honest cross-scheme number; A says how much chemistry each
  scheme's density actually encodes. Report both.

Fidelity per channel is also reported -- corr(predicted channel mass, true channel count) --
which is what caught S/Cl/Br/I carrying no signal at all in §12b.

Usage:
    python scripts/adhoc_analysis/poc2mol_scheme_discrimination.py \
        --run_dir logs/poc2mol_v2_ch11/runs/2026-08-10_21-39-12 \
        --checkpoint <ckpt> [--max_batches 20] [--output x.json]
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
from omegaconf import OmegaConf  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

from src.utils.likelihood_eval import evaluate_likelihood_ranking  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Same acceptor definition as scripts/regenerate_*_parquet.py. It MUST match, or the counts
# scored here describe a different HBA channel from the one the model was trained on.
ACCEPTOR = Chem.MolFromSmarts(
    "[$([O,S;H1;v2]),$([O,S;H0;v2]),$([N;v3;!$(N-*=[O,N,P,S])]),$([nH0,o,s;+0])]"
)


def channel_counts(mol, channels, catch_all_last=True):
    """Heavy-atom count per ligand channel, mirroring UnifiedView.match.

    Channels may overlap (HBA overlaps N_noH/O_noH by design), so this is deliberately not a
    partition -- an atom is counted in every channel it belongs to, exactly as the voxeliser
    writes it into every such channel.
    """
    n_ch = len(channels)
    counts = np.zeros(n_ch)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]          # heavy atoms only
    aromatic = [a.GetIsAromatic() for a in mol.GetAtoms()]
    n_hs = [a.GetTotalNumHs() + sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() == 1)
            for a in mol.GetAtoms()]
    acceptor = np.zeros(len(symbols), dtype=bool)
    for (idx,) in mol.GetSubstructMatches(ACCEPTOR):
        if idx < len(acceptor):
            acceptor[idx] = True

    keys = sorted(channels.keys(), key=int)
    for pos, key in enumerate(keys):
        elements = list(channels[key])
        last = catch_all_last and pos == len(keys) - 1
        for i, sym in enumerate(symbols):
            if last:
                # inverted: anything NOT in the listed elements. H is listed, so H is excluded
                # -- and H is a heavy-atom-free list here anyway.
                hit = sym not in elements
            elif "*" in elements:
                hit = True
            elif "C_aliphatic" in elements:
                hit = sym == "C" and not aromatic[i]
            elif "C_aromatic" in elements:
                hit = sym == "C" and aromatic[i]
            elif "N_withH" in elements:
                hit = sym == "N" and n_hs[i] > 0
            elif "N_noH" in elements:
                hit = sym == "N" and n_hs[i] == 0
            elif "O_withH" in elements:
                hit = sym == "O" and n_hs[i] > 0
            elif "O_noH" in elements:
                hit = sym == "O" and n_hs[i] == 0
            elif "HBA" in elements:
                hit = bool(acceptor[i])
            else:
                hit = sym in elements
            if hit:
                counts[pos] += 1
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="run dir holding resolved_config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run_dir, "resolved_config.yaml"))

    # ---- model ------------------------------------------------------------------------
    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # Same convention as Poc2MolInferenceBuilder: a Lightning checkpoint nests the UNet under
    # "model.", a bare state_dict does not. strict=True in both branches -- a silent partial
    # load here would mean scoring a half-initialised model, which is exactly the kind of
    # thing that looks like a plausible result.
    if "state_dict" in ckpt:
        inner = {k.replace("model.", "", 1): v
                 for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        model.model.load_state_dict(inner)
    else:
        model.load_state_dict(ckpt)
    model.eval().to(args.device)

    # ---- data -------------------------------------------------------------------------
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("fit")
    loader = dm.val_dataloader() if args.split == "val" else dm.test_dataloader()
    if isinstance(loader, (list, tuple)):
        loader = loader[0]

    channels = OmegaConf.to_container(cfg.data.config.ligand_channels, resolve=True)
    catch_all = bool(cfg.data.config.get("ligand_last_channel_is_catch_all", True))
    n_prot = len(cfg.data.config.protein_channels) if cfg.data.config.has_protein else 0
    print(f"scheme: {len(channels)} ligand channels, {n_prot} protein channels")

    masses, smiles_all = [], []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            batch = dm.on_after_batch_transfer(batch)
            # Training runs under `precision: bf16-mixed`, i.e. Lightning autocasts the
            # forward. Reproduce that rather than casting the grid to float32 -- the model
            # should be evaluated in the numeric regime it was trained in.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(batch["protein"])
            dens = out["predicted_ligand_voxels"].float()
            masses.append(dens.sum(dim=(2, 3, 4)).cpu().numpy())
            smiles_all.extend(batch["smiles"])
            print(f"  batch {bi+1}: {len(smiles_all)} pockets", flush=True)

    mass = np.concatenate(masses, axis=0)                       # (P, C)
    panel = sorted(set(smiles_all))
    print(f"\n{mass.shape[0]} pockets, {len(panel)} unique candidate ligands")

    cand = {}
    for s in panel:
        m = Chem.MolFromSmiles(s)
        cand[s] = channel_counts(m, channels, catch_all) if m is not None else None
    valid = np.array([cand[s] is not None for s in panel])
    C = mass.shape[1]
    cand_mat = np.stack([cand[s] if cand[s] is not None else np.zeros(C) for s in panel])

    true_counts = np.stack([cand[s] if cand[s] is not None else np.zeros(C)
                            for s in smiles_all])

    # ---- calibration: occupancy mass -> atom-count units, per channel -----------------
    denom = (true_counts * true_counts).sum(axis=0)
    scale = np.where(denom > 0, (mass * true_counts).sum(axis=0) / np.maximum(denom, 1e-9), 0.0)
    calibrated = mass / np.where(scale > 0, scale, 1.0)

    # ---- fidelity per channel ---------------------------------------------------------
    print("\n--- fidelity: corr(predicted channel mass, true channel count) ---")
    fidelity = {}
    for c in range(C):
        if true_counts[:, c].std() < 1e-9 or mass[:, c].std() < 1e-9:
            fidelity[c] = None
            print(f"  ch{c:<3} (constant, no signal measurable)")
            continue
        r = float(np.corrcoef(mass[:, c], true_counts[:, c])[0, 1])
        fidelity[c] = r
        print(f"  ch{c:<3} r = {r:+.3f}   scale = {scale[c]:.1f} mass/atom")

    # ---- positives, matched by SMILES identity (§5.1: 27 systems share a SMILES) -------
    idx = {s: i for i, s in enumerate(panel)}
    positive = np.zeros((mass.shape[0], len(panel)), dtype=bool)
    for row, s in enumerate(smiles_all):
        positive[row, idx[s]] = True

    results = {"n_pockets": int(mass.shape[0]), "n_candidates": int(len(panel)),
               "n_channels": int(C), "fidelity_per_channel_r": fidelity,
               "scale_per_channel": scale.tolist()}

    # A. composition (per-scheme): negative Euclidean distance in atom-count space
    comp = -np.linalg.norm(calibrated[:, None, :] - cand_mat[None, :, :], axis=2)
    results["composition"] = evaluate_likelihood_ranking(comp, positive, valid)

    # B. size only (scheme-independent): total heavy-atom count
    est = calibrated.sum(axis=1)
    size = -np.abs(est[:, None] - cand_mat.sum(axis=1)[None, :])
    results["size_only"] = evaluate_likelihood_ranking(size, positive, valid)

    print("\n=== discrimination (true ligand vs all other ligands in the split) ===")
    for name in ("size_only", "composition"):
        r = results[name]
        print(f"{name:<14} raw {r['likelihood_auc_raw']:.4f}   "
              f"znorm {r['likelihood_auc_znorm']:.4f}   "
              f"blind {r['likelihood_auc_pocket_blind']:.4f}")
    print("\nblind must sit at ~0.500; if not, the matrix is malformed.")
    print("size_only is the CROSS-SCHEME comparable number; composition is per-scheme "
          "(11 channels give it more descriptors than 9, independent of density quality).")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
