"""Does the RL policy still generate a DISTRIBUTION, or has it collapsed to one good scaffold?

`val/poc2mol/tanimoto` is a mean similarity to the true ligand, and it is gameable in exactly
the way RL tends to game things: a policy that ignores the pocket and always emits the most
generic drug-like scaffold in HiQBind scores a respectable mean, because drug-like molecules
resemble each other. Nothing currently logged would notice. `calculate_uniqueness` and
`calculate_novelty` are imported by `src/models/vox2smiles.py` and never called; VAL_METRICS is
loss / accuracy / validity / exact_match / tanimoto.

This measures the two things that separate "learned to condition on the pocket" from "found a
scaffold that scores well everywhere":

SPECIFICITY (the decisive one)
    tanimoto_own    each generation against ITS OWN pocket's true ligand
    tanimoto_other  the same generations against a DERANGED assignment of true ligands, so
                    every molecule is scored against some other pocket's answer
    specificity = own - other
    A pocket-conditioned model has a large positive gap. A collapsed model has own ~ other:
    still similar to everything, no longer about this pocket. This is the generation-side twin
    of the `likelihood_auc_pocket_blind` control already in likelihood_eval.py, which must land
    at 0.5 for a well-formed matrix.

DIVERSITY
    across_pocket_selfsim  mean pairwise similarity between the greedy generations for
                           DIFFERENT pockets. -> 1.0 means the same molecule everywhere.
    within_pocket_selfsim  sample n_samples per pocket at temperature and take the mean pairwise
                           similarity inside each pocket. -> 1.0 means the policy is a point mass
                           and the sampling temperature buys nothing.
    uniqueness             fraction of distinct canonical SMILES among the greedy generations
    scaffold_fraction      distinct Bemis-Murcko scaffolds / n_pockets

Read specificity FIRST. High tanimoto with collapsed specificity is not a better model, it is a
reward-hacked one, and it would make every tanimoto number in CLAUDE.md §0 meaningless.

Usage:
  ./venvPlixer/bin/python scripts/adhoc_analysis/generation_diversity.py \
      --run_dir logs/rl_dpo_const_lr2e5/runs/<ts> \
      --checkpoints sft=/path/a.ckpt,dpo2e5=/path/b.ckpt \
      --n_pockets 100 --n_samples 8
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
from rdkit.Chem.Scaffolds import MurckoScaffold                     # noqa: E402

RDLogger.DisableLog("rdApp.*")


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


def mean_pairwise(fps):
    """Mean pairwise Tanimoto within a list. Invalid entries count as 0, matching
    `paired_similarities`, so a policy cannot look diverse by emitting garbage."""
    n = len(fps)
    if n < 2:
        return float("nan")
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += similarity(fps[i], fps[j])
            count += 1
    return total / count


def scaffold_of(smiles):
    if not smiles:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return None


def derange(n, rng):
    """A permutation with no fixed point, so no molecule is ever scored against its own pocket.
    A plain shuffle leaves ~1 expected fixed point, which would bias `tanimoto_other` upward by
    exactly the effect being measured."""
    if n < 2:
        return np.arange(n)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


@torch.no_grad()
def evaluate(model, datamodule, loader, device, n_pockets, n_samples, temperature, rng):
    greedy, references = [], []
    sampled_per_pocket = []

    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    with autocast:
        for batch in loader:
            if len(greedy) >= n_pockets:
                break
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch = datamodule.on_after_batch_transfer(batch)
            pixel_values, _ = model.build_pixel_values(batch, training=False)

            take = min(n_pockets - len(greedy), pixel_values.size(0))
            pixel_values = pixel_values[:take]

            # Greedy: exactly what val/poc2mol/tanimoto scores, and the setting under which an
            # unchanged policy is deterministic (CLAUDE.md §0).
            greedy.extend(model.generate_smiles(pixel_values, max_attempts=1))
            references.extend([
                s.replace("[BOS]", "").replace("[EOS]", "")
                for s in batch["smiles_str"][:take]
            ])

            # Sampled: within-pocket spread. Greedy cannot show it -- one molecule per pocket.
            if n_samples > 1:
                repeated = pixel_values.repeat_interleave(n_samples, dim=0)
                drawn = model.generate_smiles(
                    repeated, max_attempts=1, do_sample=True, temperature=temperature
                )
                for i in range(take):
                    sampled_per_pocket.append(drawn[i * n_samples:(i + 1) * n_samples])

    greedy_fps = [fingerprint(s) for s in greedy]
    reference_fps = [fingerprint(s) for s in references]
    n = len(greedy)

    own = [similarity(greedy_fps[i], reference_fps[i]) for i in range(n)]
    perm = derange(n, rng)
    other = [similarity(greedy_fps[i], reference_fps[perm[i]]) for i in range(n)]

    valid = [s for s in greedy if s and Chem.MolFromSmiles(s) is not None]
    canonical = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in valid}
    scaffolds = {sc for sc in (scaffold_of(s) for s in valid) if sc}

    within = [mean_pairwise([fingerprint(s) for s in group]) for group in sampled_per_pocket]
    within = [w for w in within if not np.isnan(w)]

    return {
        "n_pockets": n,
        "validity": len(valid) / max(n, 1),
        "uniqueness": len(canonical) / max(len(valid), 1),
        "scaffold_fraction": len(scaffolds) / max(len(valid), 1),
        "tanimoto_own": float(np.mean(own)),
        "tanimoto_other": float(np.mean(other)),
        "specificity": float(np.mean(own) - np.mean(other)),
        "across_pocket_selfsim": mean_pairwise(greedy_fps),
        "within_pocket_selfsim": float(np.mean(within)) if within else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoints", required=True,
                        help="comma-separated name=path pairs")
    parser.add_argument("--n_pockets", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(os.path.join(args.run_dir, "resolved_config.yaml"))
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    loaders = datamodule.val_dataloader()
    loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders

    results = {}
    for entry in args.checkpoints.split(","):
        if not entry:
            continue
        name, path = entry.split("=", 1)
        model = hydra.utils.instantiate(cfg.model)
        state = torch.load(path, map_location="cpu")
        incompatible = model.load_state_dict(state.get("state_dict", state), strict=False)
        # Only the DECODER keys are structural. A decoder-only checkpoint (e.g. the pre-RL
        # stage-3 policy) carries no `poc2mol.*` at all, and that is correct rather than a
        # mismatch: the module loads the frozen upstream itself from poc2mol_ckpt_path in
        # __init__, so those weights are already right before the state_dict is applied.
        structural = [k for k in list(incompatible.missing_keys) + list(incompatible.unexpected_keys)
                      if k.startswith("model.")]
        if structural:
            raise RuntimeError(f"{path}: architecture mismatch ({structural[:4]})")
        model = model.eval().to(args.device)

        rng = np.random.default_rng(args.seed)
        results[name] = evaluate(model, datamodule, loader, args.device,
                                 args.n_pockets, args.n_samples, args.temperature, rng)
        del model
        torch.cuda.empty_cache()

        r = results[name]
        print(f"\n=== {name}  ({r['n_pockets']} pockets, {args.n_samples} samples/pocket "
              f"@ T={args.temperature})")
        print(f"  tanimoto_own           {r['tanimoto_own']:.4f}   <- the reward / logged metric")
        print(f"  tanimoto_other         {r['tanimoto_other']:.4f}   <- deranged pockets")
        print(f"  SPECIFICITY (own-other){r['specificity']:+.4f}   <- collapses to ~0 if mode-collapsed")
        print(f"  across_pocket_selfsim  {r['across_pocket_selfsim']:.4f}   <- 1.0 = same molecule every pocket")
        print(f"  within_pocket_selfsim  {r['within_pocket_selfsim']:.4f}   <- 1.0 = point mass")
        print(f"  uniqueness             {r['uniqueness']:.4f}")
        print(f"  scaffold_fraction      {r['scaffold_fraction']:.4f}")
        print(f"  validity               {r['validity']:.4f}", flush=True)

    if len(results) > 1:
        print("\n=== deltas vs the first checkpoint listed")
        base = next(iter(results))
        for name, r in list(results.items())[1:]:
            for key in ("tanimoto_own", "specificity", "across_pocket_selfsim",
                        "within_pocket_selfsim", "uniqueness", "scaffold_fraction"):
                print(f"  {name} - {base}  {key:22s} {r[key] - results[base][key]:+.4f}")

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
