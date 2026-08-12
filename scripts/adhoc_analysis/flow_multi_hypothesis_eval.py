"""Decode SEVERAL voxel hypotheses per pocket and aggregate -- the generative-only test.

The regression Poc2Mol emits one density per pocket, so the pipeline gets one shot. A flow
model defines a distribution, so it can emit N independent hypotheses and the downstream
decoder can be run on each. That is the capability reconstruction Dice cannot express and
the one that could make the generative model worth its cost:

  Tanimoto      decode each hypothesis to SMILES, compare each against the true ligand.
                `max` over hypotheses is what a pipeline that screens candidates gets;
                `mean` is what a single random draw gets on average. If max >> mean, the
                extra draws are buying real coverage rather than repeating one answer.

  Likelihood    score every candidate SMILES (1 true binder + decoys) under EVERY
  ranking       hypothesis, then aggregate per candidate before ranking. `max` asks "does
                some hypothesis explain this molecule", `mean` asks "do the hypotheses
                agree on it". AUC is computed with the project's own
                `evaluate_likelihood_ranking`, so the z-normalisation that removes the
                ligand-size nuisance is identical to the training-time metric.

Both are reported against the single-hypothesis baselines: one sample (what a naive
generative pipeline does) and the conditional-mean readout (what scores best on Dice).

Usage:
    ./venvPlixer/bin/python scripts/adhoc_analysis/flow_multi_hypothesis_eval.py \\
        --decoder_ckpt <s3 ckpt> --flow_ckpt <stage B ckpt> --n_hypotheses 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import hydra  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, DataStructs  # noqa: E402

from flow_sampling_watchdog import load_flow_checkpoint  # noqa: E402
from src.models.vox2smiles import VoxToSmilesModel  # noqa: E402
from src.utils.likelihood_eval import evaluate_likelihood_ranking  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def tanimoto(smiles_a: str, smiles_b: str) -> float:
    """Morgan/ECFP4 Tanimoto; 0.0 if either SMILES does not parse."""
    ma, mb = Chem.MolFromSmiles(smiles_a or ""), Chem.MolFromSmiles(smiles_b or "")
    if ma is None or mb is None:
        return 0.0
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
    return float(DataStructs.TanimotoSimilarity(fa, fb))


@torch.no_grad()
def score_candidates(decoder, pixel_values, cand_ids, pad_id):
    """Mean per-token log-likelihood of every candidate under every row of pixel_values.

    Mirrors ``VoxToSmilesModel._accumulate_likelihood_rows`` exactly -- same masking, same
    per-token mean -- so the numbers are comparable with the training-time metric.
    """
    masked = cand_ids.clone()
    masked[masked == pad_id] = -100
    gather_idx = masked.clone()
    gather_idx[gather_idx == -100] = 0
    valid = masked != -100
    seq_len = valid.sum(dim=1).clamp(min=1)

    rows = []
    for i in range(pixel_values.size(0)):
        repeated = pixel_values[i: i + 1].repeat(
            cand_ids.size(0), *([1] * (pixel_values.dim() - 1))
        )
        logits = decoder(repeated, labels=cand_ids).logits
        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        token_lp = log_probs.gather(-1, gather_idx.unsqueeze(-1)).squeeze(-1) * valid
        rows.append((token_lp.sum(dim=1) / seq_len).cpu().numpy())
    return np.stack(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decoder_ckpt", required=True)
    parser.add_argument("--flow_ckpt", required=True)
    parser.add_argument("--experiment", default="s3_flow_11ch")
    parser.add_argument("--n_hypotheses", type=int, default=8)
    parser.add_argument("--n_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.dirname(os.path.dirname(_HERE))
    os.environ.setdefault("PROJECT_ROOT", root)

    flow, flow_step = load_flow_checkpoint(args.flow_ckpt, device)
    decoder = VoxToSmilesModel.load_from_checkpoint(
        args.decoder_ckpt, map_location="cpu").to(device).eval()
    tokenizer = decoder.tokenizer
    print(f"decoder {args.decoder_ckpt}\nflow    {args.flow_ckpt} (step {flow_step})")

    with hydra.initialize_config_dir(version_base="1.3",
                                     config_dir=os.path.join(root, "configs")):
        cfg = hydra.compose(config_name="train", overrides=[
            f"experiment={args.experiment}", "data.num_workers=4",
            f"data.config.batch_size={args.batch_size}",
            f"data.poc2mol_ckpt_path={args.flow_ckpt}",
            "paths.output_dir=/tmp/mh", "paths.img_save_dir=/tmp/mh/img"])
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")

    # The ROC-AUC validation set is the one carrying decoys and a binder index.
    loaders = datamodule.val_dataloader()
    names = datamodule.val_dataset_names
    idx = next((i for i, n in enumerate(names) if "roc" in n or "auc" in n), 0)
    print(f"using validation set '{names[idx]}' for ranking\n")

    builder = datamodule.voxel_builder
    tani_per_hyp, score_stack, binder_idx, true_smiles_all = [], [], [], []

    for b, batch in enumerate(loaders[idx]):
        if b >= args.n_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        builder._bind(device)
        grid = builder._voxelizer(batch["atom_xyz"], batch["atom_radius"],
                                  batch["atom_slot"], int(batch["batch_size"]),
                                  int(batch["n_channels"]))
        protein = grid[:, : builder.n_protein_channels].float()
        n = protein.shape[0]
        cand_ids = batch["candidate_tokens"]["input_ids"].to(device)
        true_smiles = [s.replace(tokenizer.bos_token, "").replace(tokenizer.eos_token, "")
                       for s in batch["smiles_str"]]

        per_hyp_tani = np.zeros((args.n_hypotheses, n))
        per_hyp_scores = np.zeros((args.n_hypotheses, n, cand_ids.size(0)))
        for h in range(args.n_hypotheses):
            gen = torch.Generator(device=device)
            gen.manual_seed(args.seed + 1000 * h + b)
            density = flow.sample(protein=protein, n_steps=args.steps,
                                  guidance_scale=args.guidance, generator=gen)
            pixel = density.to(next(decoder.parameters()).dtype)
            smiles = decoder.generate_smiles(pixel, max_length=200)
            per_hyp_tani[h] = [tanimoto(s or "", t) for s, t in zip(smiles, true_smiles)]
            per_hyp_scores[h] = score_candidates(decoder, pixel, cand_ids,
                                                 tokenizer.pad_token_id)

        tani_per_hyp.append(per_hyp_tani)
        score_stack.append(per_hyp_scores)
        binder_idx.extend(int(i) for i in batch["binder_indices"])
        true_smiles_all.extend(true_smiles)
        print(f"  batch {b}: {n} pockets, {args.n_hypotheses} hypotheses each", flush=True)

    tani = np.concatenate(tani_per_hyp, axis=1)          # (H, P)
    scores = np.concatenate(score_stack, axis=1)         # (H, P, C)
    n_pockets = tani.shape[1]
    positive = np.zeros((n_pockets, scores.shape[2]), dtype=bool)
    for p, bi in enumerate(binder_idx):
        positive[p, bi] = True

    print(f"\n{n_pockets} pockets x {args.n_hypotheses} hypotheses\n")
    print("TANIMOTO of generated SMILES vs the true ligand")
    print(f"  single hypothesis (mean over draws) : {tani.mean():.4f}")
    print(f"  best-of-{args.n_hypotheses} per pocket           : {tani.max(axis=0).mean():.4f}")
    print(f"  first hypothesis only               : {tani[0].mean():.4f}")

    print("\nLIKELIHOOD RANKING (project z-norm AUC), aggregating hypotheses per candidate")
    results = {"n_pockets": int(n_pockets), "n_hypotheses": int(args.n_hypotheses),
               "tanimoto": {"mean_over_hypotheses": float(tani.mean()),
                            "best_of_n": float(tani.max(axis=0).mean()),
                            "single": float(tani[0].mean())}, "ranking": {}}
    for label, agg in (("single (h=0)", scores[0]),
                       ("mean over hypotheses", scores.mean(axis=0)),
                       ("max over hypotheses", scores.max(axis=0))):
        metrics = evaluate_likelihood_ranking(agg, positive)
        results["ranking"][label] = {k: float(v) for k, v in metrics.items()}
        print(f"  {label:>22}: znorm {metrics.get('likelihood_auc_znorm', float('nan')):.4f}  "
              f"raw {metrics.get('likelihood_auc_raw', float('nan')):.4f}  "
              f"blind {metrics.get('likelihood_auc_pocket_blind', float('nan')):.4f}")
    print("\n'blind' is the sanity control and must sit at ~0.5; if it does not, the matrix "
          "is malformed and the AUCs mean nothing.")

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
