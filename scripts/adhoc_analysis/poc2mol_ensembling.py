"""Factorial test of ensembling axes on likelihood ranking, with all members saved.

Three axes:
  AUGMENTATION  re-voxelise each pocket under a fresh random rotation/translation. ⚠️ The
                Poc2Mol val set is DETERMINISTIC by default (§3e: rotate false, translation 0)
                so this axis is a no-op unless --rotate is passed. A previous version of this
                script silently measured nothing because of that -- the tell was
                model_ensemble == aug_x_model to 16 digits, only possible if the replicates
                were bitwise identical.
  MODEL         average across Poc2Mol checkpoints.
  MASS x DECODER  handled separately (§12d): the two readouts are near-orthogonal
                (within-pocket r = +0.024) and their blend reached 0.785.

NORMALISATION -- the thing that makes an ensemble honest. Each member's (pocket x candidate)
matrix is COLUMN z-normalised before averaging, which is the same transform
`likelihood_auc_znorm` applies. Two consequences:
  * a member with a wide dynamic range cannot dominate the average -- every member's columns
    have unit sd, so all contribute equally;
  * the single-member baseline is computed on the SAME normalised matrices, so the reported
    ensemble gain is not contaminated by the normalisation itself.
The earlier version row-standardised before averaging while the metric z-normalised by column,
which inflated every ensemble number by ~+0.005 relative to its baseline.

All member matrices are written to an .npz so subset selection can be done post hoc without
re-running the forwards.

Usage:
  python scripts/adhoc_analysis/poc2mol_ensembling.py --run_dir <dir> \
      --checkpoints a.ckpt,b.ckpt --n_aug 4 --rotate --output ens.json --save_matrices ens.npz
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import hydra                                                        # noqa: E402
from omegaconf import OmegaConf, open_dict                          # noqa: E402
from rdkit import Chem, RDLogger                                    # noqa: E402
from src.utils.likelihood_eval import per_pocket_auc, znormalise_columns   # noqa: E402
from scripts.adhoc_analysis.poc2mol_scheme_discrimination import channel_counts  # noqa: E402
RDLogger.DisableLog("rdApp.*")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--n_aug", type=int, default=4)
    p.add_argument("--rotate", action="store_true",
                   help="enable rotation/translation on the val set so the augmentation axis is real")
    p.add_argument("--translation", type=float, default=6.0)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    p.add_argument("--save_matrices", default=None)
    args = p.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run_dir, "resolved_config.yaml"))
    channels = OmegaConf.to_container(cfg.data.config.ligand_channels, resolve=True)
    catch_all = bool(cfg.data.config.get("ligand_last_channel_is_catch_all", True))
    ckpts = [c for c in args.checkpoints.split(",") if c]

    if args.rotate:
        with open_dict(cfg):
            for key in ("val_dataset",):
                if key in cfg.data and cfg.data[key] is not None:
                    cfg.data[key].rotate = True
                    cfg.data[key].translation = args.translation
        print(f"augmentation ENABLED on val: rotate=True translation={args.translation}")
    else:
        print("augmentation DISABLED (val is deterministic) -- the aug axis will be a no-op")

    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("fit")
    loader = dm.val_dataloader()
    if isinstance(loader, (list, tuple)):
        loader = loader[0]

    def load(ck):
        m = hydra.utils.instantiate(cfg.model)
        s = torch.load(ck, map_location="cpu")
        if "state_dict" in s:
            m.model.load_state_dict({k.replace("model.", "", 1): v
                                     for k, v in s["state_dict"].items() if k.startswith("model.")})
        else:
            m.load_state_dict(s)
        return m.eval().to(args.device)

    mass, order = {}, None
    for ci, ck in enumerate(ckpts):
        model = load(ck)
        for a in range(args.n_aug):
            # Same seed across models at a given aug index, so model-vs-model differences are
            # not contaminated by different augmentations.
            torch.manual_seed(4242 + a); np.random.seed(4242 + a)
            rows, smi = [], []
            with torch.no_grad():
                for bi, batch in enumerate(loader):
                    if args.max_batches and bi >= args.max_batches:
                        break
                    batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    batch = dm.on_after_batch_transfer(batch)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(batch["protein"])
                    rows.append(out["predicted_ligand_voxels"].float().sum(dim=(2, 3, 4)).cpu().numpy())
                    smi.extend(batch["smiles"])
            mass[(ci, a)] = np.concatenate(rows, 0)
            if order is None:
                order = smi
            elif smi != order:
                raise RuntimeError("pocket order changed between replicates -- not alignable")
            print(f"  ckpt{ci} aug{a}: {mass[(ci,a)].shape[0]} pockets", flush=True)
        del model; torch.cuda.empty_cache()

    # sanity: did augmentation actually vary anything?
    if args.n_aug > 1:
        d = np.abs(mass[(0, 0)] - mass[(0, 1)]).max()
        print(f"\nmax |mass(aug0) - mass(aug1)| for ckpt0 = {d:.4f}"
              f"{'  ⚠️ ZERO -- augmentation is not varying' if d == 0 else ''}")

    panel = sorted(set(order))
    cand = {}
    for s in panel:
        m = Chem.MolFromSmiles(s)
        cand[s] = channel_counts(m, channels, catch_all) if m is not None else None
    valid = np.array([cand[s] is not None for s in panel])
    C = next(iter(mass.values())).shape[1]
    cmat = np.stack([cand[s] if cand[s] is not None else np.zeros(C) for s in panel])
    truec = np.stack([cand[s] if cand[s] is not None else np.zeros(C) for s in order])
    idx = {s: i for i, s in enumerate(panel)}
    pos = np.zeros((len(order), len(panel)), dtype=bool)
    for r, s in enumerate(order):
        pos[r, idx[s]] = True

    def comp_matrix(mv):
        den = (truec * truec).sum(axis=0)
        sc = np.where(den > 0, (mv * truec).sum(axis=0) / np.maximum(den, 1e-9), 0.0)
        cal = mv / np.where(sc > 0, sc, 1.0)
        return -np.linalg.norm(cal[:, None, :] - cmat[None, :, :], axis=2)

    # Column z-normalise EVERY member once. From here on all scores are on a common scale, so
    # averaging is unweighted and no member can dominate through dynamic range.
    keys = sorted(mass.keys())
    Z = {k: znormalise_columns(comp_matrix(mass[k])) for k in keys}

    def auc(m):
        return per_pocket_auc(m, pos, valid)[0]

    def ens(ks):
        return auc(np.mean([Z[k] for k in ks], axis=0))

    M, R = len(ckpts), args.n_aug
    singles = [auc(Z[k]) for k in keys]
    res = {
        "n_pockets": int(len(order)), "n_candidates": int(len(panel)),
        "rotate": bool(args.rotate), "n_models": M, "n_aug": R,
        "single_mean": float(np.mean(singles)), "single_std": float(np.std(singles)),
        "single_min": float(np.min(singles)), "single_max": float(np.max(singles)),
        "aug_ensemble": float(np.mean([ens([(ci, a) for a in range(R)]) for ci in range(M)])),
        "model_ensemble": float(np.mean([ens([(ci, a) for ci in range(M)]) for a in range(R)])),
        "aug_x_model": float(ens(keys)),
    }

    print("\n=== composition readout, per-member column z-norm (equal weight) ===")
    b = res["single_mean"]
    print(f"  single member        {b:.4f}  (sd {res['single_std']:.4f}, "
          f"range {res['single_min']:.4f}-{res['single_max']:.4f})")
    for lab, k in (("+ augmentation (x%d)" % R, "aug_ensemble"),
                   ("+ model (x%d)" % M, "model_ensemble"),
                   ("+ both (%dx%d)" % (M, R), "aug_x_model")):
        print(f"  {lab:<21}{res[k]:.4f}   delta {res[k]-b:+.4f}")

    if args.save_matrices:
        np.savez_compressed(
            args.save_matrices,
            members=np.stack([Z[k] for k in keys]),
            member_keys=np.array([f"ckpt{c}_aug{a}" for c, a in keys]),
            checkpoints=np.array(ckpts), positive=pos, valid=valid,
            panel=np.array(panel), pocket_smiles=np.array(order),
        )
        print(f"\nsaved {len(keys)} member matrices -> {args.save_matrices}")

    if args.output:
        json.dump(res, open(args.output, "w"), indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
