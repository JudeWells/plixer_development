"""Stage-1 readout: whole-SMILES exact match under free-running generation.

Why this script exists
----------------------
Token accuracy and validity both saturate long before the decoder is actually reconstructing
molecules -- by step 67k the production runs sat at 0.992 token accuracy and 0.999 validity,
neither of which discriminates any further. Whole-molecule exact match was 63% at the same
point and still moving. It is the metric to track (CLAUDE.md 9d).

It is also the check that free-running behaviour matches the teacher-forced metrics. A
teacher-forced number can be inflated by a padding or off-by-one bug; exact match cannot,
because generation never sees the target. If token accuracy is high and exact match is
near zero, distrust the teacher-forced path.

Usage
-----
    python evaluations/evaluate_stage1_exact_match.py \
        --checkpoint logs/exp1_s1_prod_sumagg/runs/<stamp>/checkpoints/last.ckpt \
        --voxel_aggregation sum --n_batches 24

    # conditioning ablation: is the model actually reading the voxels?
    python evaluations/evaluate_stage1_exact_match.py --checkpoint ... --zero_voxels

Note this measures the STAGE-1 task -- ground-truth ligand voxels, no pocket. Stage 3
conditions on Poc2Mol's predicted density and is much harder, so do not carry these numbers
forward as an expectation for the end-to-end model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

from src.utils.metrics import calculate_exact_match, calculate_validity  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def canonical(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="path to a .ckpt")
    p.add_argument("--experiment", default="exp1_zinc_protein",
                   help="Hydra experiment config; must match the checkpoint's channel count")
    p.add_argument("--voxel_aggregation", default="max", choices=["max", "sum"],
                   help="must match what the checkpoint was TRAINED with, or the input "
                        "distribution differs from training and the numbers are meaningless")
    p.add_argument("--voxel_radius_scale", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_batches", type=int, default=24, help="~n_batches*batch_size molecules")
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--max_length", type=int, default=200)
    p.add_argument("--zero_voxels", action="store_true",
                   help="ablation: replace the input with zeros. Whatever accuracy survives "
                        "is the unconditional SMILES prior, not conditioning.")
    p.add_argument("--output", default=None, help="write results as JSON here")
    return p.parse_args()


def build(args):
    """Compose the config and instantiate datamodule + model, mirroring training exactly."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("PROJECT_ROOT", root)
    with hydra.initialize_config_dir(version_base="1.3",
                                     config_dir=os.path.join(root, "configs")):
        cfg = hydra.compose(config_name="train", overrides=[
            f"experiment={args.experiment}",
            f"data.num_workers={args.num_workers}",
            f"data.config.batch_size={args.batch_size}",
            "data.config.target_samples_per_batch=512",
            f"data.config.voxel_aggregation={args.voxel_aggregation}",
            f"data.config.voxel_radius_scale={args.voxel_radius_scale}",
        ])
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    # strict=False: checkpoints carry MeanMetric buffers that come and go as metrics are
    # added. Missing/unexpected keys are reported so a genuine architecture mismatch is
    # still visible rather than silently tolerated.
    incompatible = model.load_state_dict(ckpt["state_dict"], strict=False)
    unexpected = [k for k in incompatible.unexpected_keys if not k.startswith("val_")]
    missing = [k for k in incompatible.missing_keys if not k.startswith("val_")]
    if unexpected or missing:
        print(f"WARNING: state_dict mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
    return datamodule, model.cuda().eval(), ckpt.get("global_step")


def main():
    args = parse_args()
    datamodule, model, global_step = build(args)
    tokenizer = model.tokenizer

    loader = datamodule.val_dataloader()
    if isinstance(loader, (list, tuple)):
        loader = loader[0]

    generated, references = [], []
    tf_correct = tf_total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.n_batches:
                break
            # Voxels are built in on_after_batch_transfer, not in the dataset -- the
            # dataloader yields CPU atom records (CLAUDE.md 3b).
            batch = datamodule.on_after_batch_transfer(batch, 0)
            pixel_values = batch["pixel_values"].cuda()
            labels = batch["input_ids"].cuda()
            if args.zero_voxels:
                pixel_values = torch.zeros_like(pixel_values)

            masked_labels = labels.clone()
            masked_labels[masked_labels == tokenizer.pad_token_id] = -100
            outputs = model(pixel_values, labels=masked_labels)

            # Same slicing as validation_step: HF shifts decoder inputs internally, so
            # logits[:, i] aligns with labels[:, i]; start at 1 to skip the BOS position.
            shift_logits = outputs.logits[..., 1:, :]
            shift_labels = masked_labels[..., 1:]
            keep = shift_labels != -100
            tf_correct += ((shift_logits.argmax(-1) == shift_labels) & keep).sum().item()
            tf_total += keep.sum().item()

            tokens = model.model.generate(
                pixel_values,
                max_length=args.max_length,
                do_sample=False,
                decoder_start_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated.extend(s.replace(" ", "")
                             for s in tokenizer.batch_decode(tokens, skip_special_tokens=True))
            references.extend(s.replace(" ", "")
                              for s in tokenizer.batch_decode(labels, skip_special_tokens=True))

    n = len(generated)
    string_exact = sum(g == r for g, r in zip(generated, references)) / n
    results = {
        "checkpoint": args.checkpoint,
        "global_step": global_step,
        "voxel_aggregation": args.voxel_aggregation,
        "zero_voxels": args.zero_voxels,
        "n_molecules": n,
        "teacher_forced_token_accuracy": tf_correct / tf_total,
        "validity": calculate_validity(generated),
        "exact_match_string": string_exact,
        "exact_match_molecule": calculate_exact_match(generated, references),
    }

    print()
    for key, value in results.items():
        print(f"{key:32s} {value}")
    if args.zero_voxels:
        print("\n  Zeroed input: whatever accuracy remains is the unconditional SMILES")
        print("  prior. On a well-conditioned stage-1 model this collapses (0.99 -> 0.44).")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
