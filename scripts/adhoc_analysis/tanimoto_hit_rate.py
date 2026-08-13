"""Where in the Tanimoto DISTRIBUTION does DPO act -- the bulk, or the top end?

`val/poc2mol/tanimoto` is a mean, and a mean can rise three different ways: every pocket
improves a little, a few pockets improve a lot, or the bad tail is trimmed. Those have very
different value for a candidate generator. What matters in practice is the hit rate -- the
fraction of targets for which the model proposes something genuinely similar to the true
ligand -- so this reports P(Tanimoto > t) across thresholds rather than a single average.

Two regimes, and the difference between them is the interesting part:

GREEDY      one molecule per pocket, which is what `val/poc2mol/tanimoto` scores.
BEST-OF-N   n samples per pocket at temperature, scored on the best one. This is the
            deployment-relevant number for a generator (you would propose several candidates
            and screen them), and it is where the diversity collapse should bite: the DPO
            policy's within-pocket self-similarity is 0.446 against the pre-RL 0.133
            (results/e2e/diversity.json), so its n samples explore far less. A model can win
            on greedy and lose on best-of-N by having nothing left to explore with, and that
            trade would be invisible to every metric currently logged.

Reports hit rates at several thresholds, the full quantile curve, and the greedy->best-of-N
uplift per model, which is a direct read on how much the sampling distribution is still worth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import hydra                                                        # noqa: E402
from omegaconf import OmegaConf                                     # noqa: E402
from rdkit import Chem, DataStructs, RDLogger                       # noqa: E402
from rdkit.Chem import AllChem                                      # noqa: E402

RDLogger.DisableLog("rdApp.*")

THRESHOLDS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)


def fingerprint(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def similarity(a, b):
    if a is None or b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(a, b))


@torch.no_grad()
def collect(model, datamodule, loader, device, n_pockets, n_samples, temperature, chunk):
    """Per-pocket greedy similarity and the full n_samples x pocket similarity matrix."""
    greedy_sim, sampled_sim = [], []

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            if len(greedy_sim) >= n_pockets:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch = datamodule.on_after_batch_transfer(batch)
            pixel_values, _ = model.build_pixel_values(batch, training=False)

            take = min(n_pockets - len(greedy_sim), pixel_values.size(0))
            pixel_values = pixel_values[:take]
            references = [
                s.replace("[BOS]", "").replace("[EOS]", "")
                for s in batch["smiles_str"][:take]
            ]
            reference_fps = [fingerprint(s) for s in references]

            generated = model.generate_smiles(pixel_values, max_attempts=1)
            greedy_sim.extend(
                similarity(fingerprint(g), reference_fps[i]) for i, g in enumerate(generated)
            )

            # Chunked: 200 pockets x 8 samples in one generate call would be a 1600-sequence
            # batch and the KV cache for 200-token decoding does not fit alongside four
            # training arms on the same device.
            per_pocket = [[] for _ in range(take)]
            for start in range(0, take, chunk):
                stop = min(start + chunk, take)
                repeated = pixel_values[start:stop].repeat_interleave(n_samples, dim=0)
                drawn = model.generate_smiles(
                    repeated, max_attempts=1, do_sample=True, temperature=temperature
                )
                for i in range(stop - start):
                    group = drawn[i * n_samples:(i + 1) * n_samples]
                    per_pocket[start + i] = [
                        similarity(fingerprint(g), reference_fps[start + i]) for g in group
                    ]
            sampled_sim.extend(per_pocket)

    return np.array(greedy_sim), np.array(sampled_sim)


def summarise(greedy, sampled):
    best_of_n = sampled.max(axis=1) if sampled.size else np.array([])
    out = {
        "n_pockets": int(greedy.size),
        "greedy_mean": float(greedy.mean()),
        "best_of_n_mean": float(best_of_n.mean()) if best_of_n.size else float("nan"),
        "greedy_quantiles": {q: float(np.quantile(greedy, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
        "hit_greedy": {str(t): float((greedy > t).mean()) for t in THRESHOLDS},
        "hit_best_of_n": {str(t): float((best_of_n > t).mean()) if best_of_n.size else float("nan")
                          for t in THRESHOLDS},
    }
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoints", required=True, help="comma-separated name=path")
    parser.add_argument("--n_pockets", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--chunk", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run_dir, "resolved_config.yaml"))
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    loaders = datamodule.val_dataloader()
    loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders

    results, raw = {}, {}
    for entry in args.checkpoints.split(","):
        if not entry:
            continue
        name, path = entry.split("=", 1)
        model = hydra.utils.instantiate(cfg.model)
        state = torch.load(path, map_location="cpu")
        incompatible = model.load_state_dict(state.get("state_dict", state), strict=False)
        bad = [k for k in list(incompatible.missing_keys) + list(incompatible.unexpected_keys)
               if k.startswith("model.")]
        if bad:
            raise RuntimeError(f"{path}: architecture mismatch ({bad[:4]})")
        model = model.eval().to(args.device)

        greedy, sampled = collect(model, datamodule, loader, args.device,
                                  args.n_pockets, args.n_samples, args.temperature, args.chunk)
        results[name] = summarise(greedy, sampled)
        raw[name] = {"greedy": greedy.tolist(), "sampled": sampled.tolist()}
        del model
        torch.cuda.empty_cache()

        r = results[name]
        print(f"\n=== {name}  ({r['n_pockets']} pockets, best-of-{args.n_samples} @ T={args.temperature})")
        print(f"  mean   greedy {r['greedy_mean']:.4f}   best-of-n {r['best_of_n_mean']:.4f}")
        print(f"  greedy quantiles  " + "  ".join(
            f"p{int(q*100)} {v:.3f}" for q, v in r["greedy_quantiles"].items()))
        print(f"  {'thresh':>8} {'greedy':>9} {'best-of-n':>11}")
        for t in THRESHOLDS:
            print(f"  {t:>8.2f} {r['hit_greedy'][str(t)]:>9.3f} {r['hit_best_of_n'][str(t)]:>11.3f}")
        sys.stdout.flush()

    names = list(results)
    if len(names) > 1:
        base = names[0]
        print(f"\n=== hit-rate deltas vs {base}")
        print(f"  {'thresh':>8} " + "  ".join(f"{n[:14]:>16s}" for n in names[1:]))
        for t in THRESHOLDS:
            row = f"  {t:>8.2f} "
            for n in names[1:]:
                dg = results[n]["hit_greedy"][str(t)] - results[base]["hit_greedy"][str(t)]
                db = results[n]["hit_best_of_n"][str(t)] - results[base]["hit_best_of_n"][str(t)]
                row += f"  g{dg:+.3f}/b{db:+.3f}"
            print(row)

    if args.output:
        with open(args.output, "w") as handle:
            json.dump({"summary": results, "raw": raw}, handle, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
