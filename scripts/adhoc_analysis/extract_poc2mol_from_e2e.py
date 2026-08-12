"""Pull the Poc2Mol half out of an end-to-end checkpoint, in standalone Poc2Mol format.

An `EndToEndPoc2Smiles` checkpoint holds both models: `poc2mol.model.<...>` for the U-Net and
`model.<...>` for the ViT+GPT-2 decoder. Anything that wants to load the upstream on its own
-- `Poc2MolInferenceBuilder`, `density_diagnostics.py`, or `EndToEndPoc2Smiles`'s own
`poc2mol_ckpt_path` -- expects a Lightning Poc2Mol checkpoint, whose keys are `model.<...>`.
Feeding it the combined file loads garbage or fails outright, since stripping `model.` off a
combined state_dict mangles the decoder keys into the U-Net's namespace.

Why this exists: it lets the end-to-end density be evaluated on equal terms with the original
one. Train a fresh decoder on the extracted upstream, frozen, and compare against the same
decoder trained on the original frozen upstream -- the two runs then differ in exactly one
thing, which density they learn from, with no moving target in either.

Usage:
    ./venvPlixer/bin/python scripts/adhoc_analysis/extract_poc2mol_from_e2e.py \\
        --ckpt logs/e2e_b_balanced_r3/runs/.../checkpoints/last.ckpt \\
        --out checkpoints/e2e_b_r3_poc2mol.ckpt
"""
from __future__ import annotations

import argparse

import torch

PREFIX = "poc2mol.model."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="end-to-end checkpoint")
    parser.add_argument("--out", required=True, help="where to write the Poc2Mol checkpoint")
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    upstream = {
        f"model.{key[len(PREFIX):]}": value
        for key, value in state_dict.items()
        if key.startswith(PREFIX)
    }
    if not upstream:
        raise SystemExit(
            f"no {PREFIX}* keys in {args.ckpt} -- is this an end-to-end checkpoint?"
        )

    out = {
        "state_dict": upstream,
        # Carry the provenance across rather than orphaning it: a Poc2Mol file that cannot
        # say which end-to-end run produced it is a file nobody will trust in a month.
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "provenance": checkpoint.get("provenance"),
        "extracted_from": args.ckpt,
    }
    torch.save(out, args.out)
    print(f"wrote {len(upstream)} tensors to {args.out}")
    print(f"  source: {args.ckpt}")
    print(f"  global_step: {checkpoint.get('global_step')}")


if __name__ == "__main__":
    main()
