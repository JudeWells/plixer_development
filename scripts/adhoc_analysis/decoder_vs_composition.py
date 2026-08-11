"""Does the decoder's ranking add anything to a plain composition readout -- and does its
likelihood really fall with molecule size?

Two questions, one score matrix.

1. SIZE. CLAUDE.md 5.1 measured `corr(ligand mean likelihood, heavy-atom count) = -0.758` on the
   PUBLISHED decoder over the 943-ligand chrono panel. That is not evidence about this decoder on
   this panel, and the sign is not obvious a priori: a longer SMILES gives the autoregressive
   decoder more context, so late tokens get cheaper and the mean per-token log-prob could just as
   easily RISE with length. Measured here rather than assumed.

2. ORTHOGONALITY. A parameter-free readout of Poc2Mol's per-channel occupancy mass scores z-norm
   AUC 0.761, against the decoder's 0.745. Equal AUCs do not imply equal information -- they could
   rank differently and be complementary. Residualising the decoder's scores on the composition
   scores WITHIN each pocket asks what the decoder knows that the formula does not.

Usage:
    python scripts/adhoc_analysis/decoder_vs_composition.py --checkpoint <ckpt> \
        [--arrays /tmp/ceiling_arrays.npz] [--output x.json]
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
from rdkit import RDLogger  # noqa: E402

from src.utils.likelihood_eval import evaluate_likelihood_ranking, per_pocket_auc, znormalise_columns  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--arrays", default="/tmp/ceiling_arrays.npz",
                   help="npz written by poc2mol_density_ceiling.py")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--mask_protein", action="store_true",
                   help="Zero the protein channels AND the presence flag, i.e. the exact "
                        "'protein absent' condition assemble_decoder_input produces and that "
                        "the model saw for 25%% of training samples. Poc2Mol has already run "
                        "by this point, so the PREDICTED ligand density is unchanged -- this "
                        "isolates what the decoder takes from the pocket directly, as opposed "
                        "to what it takes from the predicted density.")
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("PROJECT_ROOT", root)
    with hydra.initialize_config_dir(version_base="1.3",
                                     config_dir=os.path.join(root, "configs")):
        cfg = hydra.compose(config_name="train", overrides=[
            "experiment=exp1_s3_protein", "data.num_workers=4",
            f"data.config.batch_size={args.batch_size}",
            f"data.config.val_batch_size={args.batch_size}",
            "data.config.target_samples_per_batch=32",
            "paths.output_dir=/tmp/dvc", "paths.img_save_dir=/tmp/dvc/img",
        ])
    os.makedirs("/tmp/dvc/img", exist_ok=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"],
                          strict=False)
    model = model.cuda().eval()
    tokenizer = model.tokenizer

    loader = datamodule.val_dataloader()[0]
    panel = list(datamodule.val_datasets["roc_auc_plinder"].decoy_smiles_list)

    rows, binder_indices = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
            # training=False -> fraction 1.0 -> Poc2Mol's PREDICTION, the deployed condition
            out = datamodule.voxel_builder(batch, apply_quality_filter=False,
                                           training=False, global_step=0)
            pixel_values = out["pixel_values"]
            if args.mask_protein:
                # Channels [0:9] are the (predicted) ligand density; everything after is the
                # 4 protein channels plus the constant presence flag. Zeroing from 9 onward
                # reproduces assemble_decoder_input's keep=False branch exactly.
                pixel_values = pixel_values.clone()
                pixel_values[:, 9:] = 0
            cand_ids = out["candidate_tokens"]["input_ids"].to(pixel_values.device)

            masked = cand_ids.clone()
            masked[masked == tokenizer.pad_token_id] = -100
            gather_idx = masked.clone()
            gather_idx[gather_idx == -100] = 0
            valid = masked != -100
            seq_len = valid.sum(dim=1).clamp(min=1)

            for i in range(pixel_values.size(0)):
                repeated = pixel_values[i:i + 1].repeat(
                    cand_ids.size(0), *([1] * (pixel_values.dim() - 1)))
                logits = model(repeated, labels=cand_ids).logits
                log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                token_lp = log_probs.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1) * valid
                rows.append((token_lp.sum(1) / seq_len).cpu().numpy())
            binder_indices.extend(int(x) for x in out["binder_indices"])
            # token count per candidate is identical across batches
            n_tokens = valid.sum(1).cpu().numpy()

    decoder = np.stack(rows)                      # (P, N) mean per-token log-prob
    cached = np.load(args.arrays)
    composition_counts = cached["candidate_matrix"]
    positive = np.zeros_like(decoder, dtype=bool)
    for r, b in enumerate(binder_indices):
        target = panel[b]
        positive[r] = np.array([s == target for s in panel])
    assert positive.shape == cached["positive"].shape, "pocket count mismatch vs cached arrays"
    if not np.array_equal(positive, cached["positive"]):
        print("WARNING: pocket ordering differs from the cached run; residuals may be misaligned")
    valid_columns = cached["valid_columns"]

    results = {}

    # ---- Q1: does mean per-token likelihood fall with molecule size? ----------------
    ligand_mean = decoder.mean(axis=0)            # the per-ligand effect
    heavy = composition_counts.sum(axis=1)
    keep = valid_columns
    r_size = float(np.corrcoef(ligand_mean[keep], heavy[keep])[0, 1])
    r_len = float(np.corrcoef(ligand_mean[keep], n_tokens[keep])[0, 1])
    results["corr_ligand_mean_likelihood_vs_heavy_atoms"] = r_size
    results["corr_ligand_mean_likelihood_vs_n_tokens"] = r_len
    print("\n--- Q1: per-ligand mean likelihood vs size ---")
    print(f"    corr(mean likelihood, heavy-atom count) = {r_size:+.3f}")
    print(f"    corr(mean likelihood, n tokens)         = {r_len:+.3f}")
    print("    (CLAUDE.md 5.1 measured -0.758 for the PUBLISHED decoder on the 943-ligand panel)")

    # ---- variance decomposition, as in 5.1 ------------------------------------------
    grand = decoder.mean()
    pocket_effect = decoder.mean(axis=1, keepdims=True) - grand
    ligand_effect = decoder.mean(axis=0, keepdims=True) - grand
    interaction = decoder - grand - pocket_effect - ligand_effect
    total = decoder.var()
    parts = {"pocket": float((pocket_effect ** 2).mean() / total),
             "ligand": float((ligand_effect ** 2).mean() / total),
             "interaction": float(interaction.var() / total)}
    results["variance_decomposition"] = parts
    print("\n--- variance decomposition of the decoder matrix ---")
    for k, v in parts.items():
        print(f"    {k:12s} {v*100:5.1f}%")

    # ---- Q2: what does the decoder add over composition? ----------------------------
    scale = None
    # rebuild the composition score exactly as the ceiling script did (size+composition)
    predicted_mass = cached["predicted_mass"]
    true_counts = cached["true_counts"]
    scale = predicted_mass.mean(axis=0) / np.clip(true_counts.mean(axis=0), 1e-9, None)
    calibrated = predicted_mass / np.clip(scale, 1e-9, None)[None, :]
    composition = -np.linalg.norm(calibrated[:, None, :] - composition_counts[None, :, :], axis=2)

    print("\n--- AUCs on this panel ---")
    for label, mat in [("decoder", decoder), ("composition", composition)]:
        m = evaluate_likelihood_ranking(mat, positive, valid_columns)
        results[f"auc_{label}"] = m
        print(f"    {label:12s} raw {m['likelihood_auc_raw']:.4f}   znorm {m['likelihood_auc_znorm']:.4f}")

    # residualise WITHIN each pocket: what survives after the formula is accounted for
    def residualise(target, explanator):
        out = np.zeros_like(target)
        for p in range(target.shape[0]):
            x, y = explanator[p][keep], target[p][keep]
            b, a = np.polyfit(x, y, 1)
            full = target[p] - (a + b * explanator[p])
            out[p] = full
        return out

    for label, target, explanator in [
        ("decoder | composition removed", decoder, composition),
        ("composition | decoder removed", composition, decoder),
    ]:
        resid = residualise(target, explanator)
        auc_raw, _ = per_pocket_auc(resid, positive, valid_columns)
        auc_z, _ = per_pocket_auc(znormalise_columns(resid), positive, valid_columns)
        results[f"residual_{label}"] = {"raw": auc_raw, "znorm": auc_z}
        print(f"    {label:32s} raw {auc_raw:.4f}   znorm {auc_z:.4f}")

    # how correlated are the two scorers within a pocket?
    within = [float(np.corrcoef(decoder[p][keep], composition[p][keep])[0, 1])
              for p in range(decoder.shape[0])]
    results["mean_within_pocket_corr_decoder_composition"] = float(np.mean(within))
    print(f"\n    mean within-pocket corr(decoder, composition) = {np.mean(within):+.3f}")

    suffix = "_maskprot" if args.mask_protein else ""
    np.savez(f"/tmp/decoder_matrix{suffix}.npz", decoder=decoder, composition=composition,
             positive=positive, valid_columns=valid_columns)
    if args.output:
        with open(args.output, "w") as h:
            json.dump(results, h, indent=2, default=float)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
