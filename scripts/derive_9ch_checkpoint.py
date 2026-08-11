"""Derive a 9-channel decoder checkpoint from the 14-channel one, for a fair A/B baseline.

The experiment-1 A/B asks whether protein channels help. That comparison is only attributable if
both arms START from the same place. The only genuine 9-channel stage-1 checkpoint we have is from
the pre-2026-08-07 code era that plateaued at val/loss 0.19, against the 14-channel arm's 0.019 --
using it would measure stage-1 quality, not protein channels (CLAUDE.md 10d).

The fix uses the fact that the patch embedding is a Conv3d, hence linear in its input channels:

    conv(x) = sum_c W[:, c] * x[:, c] + b

If channels 9..13 (protein + presence flag) are zero, their terms vanish, so a 14-channel model
fed zeroed protein is EXACTLY a 9-channel model whose weight is W[:, :9]. Slicing therefore gives
a 9-channel network that is functionally identical to the 14-channel one on ligand-only input --
same stage-1 training, same weights, no retraining, no confound.

Everything outside the patch embedding is untouched: the ViT blocks, the GPT-2 decoder and the
cross-attention all have channel-independent shapes.

Verified numerically here, not assumed: both models are run on the same ligand input (the
14-channel one with zeros in the protein slots) and their logits compared.

Usage:
    python scripts/derive_9ch_checkpoint.py --input checkpoints/s1_maxagg_last.ckpt \
        --output checkpoints/s1_maxagg_last_9ch.ckpt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra  # noqa: E402

CONV_KEY = "model.encoder.embeddings.patch_embeddings.projection.weight"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="checkpoints/s1_maxagg_last.ckpt")
    parser.add_argument("--output", default="checkpoints/s1_maxagg_last_9ch.ckpt")
    parser.add_argument("--n_ligand_channels", type=int, default=9)
    parser.add_argument("--skip_verify", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    weight = state_dict[CONV_KEY]
    n_in = weight.shape[1]
    print(f"patch embedding: {tuple(weight.shape)}  ({n_in} input channels)")
    if n_in == args.n_ligand_channels:
        print("already the target width; nothing to do")
        return

    sliced = dict(state_dict)
    sliced[CONV_KEY] = weight[:, : args.n_ligand_channels].clone()
    print(f"sliced to        {tuple(sliced[CONV_KEY].shape)}")

    # Keep only the tensors. Optimizer state is deliberately dropped: it is shaped for the
    # 14-channel conv, and these weights are an INITIALISATION for a new stage, not a resume.
    torch.save({"state_dict": sliced,
                "derived_from": os.path.abspath(args.input),
                "note": "patch-embedding conv sliced to the first "
                        f"{args.n_ligand_channels} (ligand) input channels"},
               args.output)
    print(f"wrote {args.output}")

    if args.skip_verify:
        return

    # ---- prove the two are the same function on ligand-only input --------------------
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("PROJECT_ROOT", root)
    with hydra.initialize_config_dir(version_base="1.3",
                                     config_dir=os.path.join(root, "configs")):
        cfg14 = hydra.compose(config_name="train", overrides=[
            "experiment=exp1_s3_protein",
            "paths.output_dir=/tmp/derive", "paths.img_save_dir=/tmp/derive/img"])
        cfg9 = hydra.compose(config_name="train", overrides=[
            "experiment=exp1_s3_baseline",
            "paths.output_dir=/tmp/derive", "paths.img_save_dir=/tmp/derive/img"])
    os.makedirs("/tmp/derive/img", exist_ok=True)

    model14 = hydra.utils.instantiate(cfg14.model)
    model14.load_state_dict(state_dict, strict=False)
    model9 = hydra.utils.instantiate(cfg9.model)
    model9.load_state_dict(sliced, strict=False)
    model14, model9 = model14.cuda().eval(), model9.cuda().eval()

    torch.manual_seed(0)
    ligand = torch.rand(2, args.n_ligand_channels, 32, 32, 32).cuda()
    padded = torch.zeros(2, n_in, 32, 32, 32).cuda()
    padded[:, : args.n_ligand_channels] = ligand
    labels = torch.randint(5, 70, (2, 12)).cuda()

    with torch.no_grad():
        a = model14(padded.to(next(model14.parameters()).dtype), labels=labels).logits.float()
        b = model9(ligand.to(next(model9.parameters()).dtype), labels=labels).logits.float()
    diff = (a - b).abs().max().item()
    print(f"\nmax |logit difference| between 14ch(protein=0) and 9ch: {diff:.3e}")
    print("PASS -- functionally identical, so the A/B arms start from the same model"
          if diff < 1e-3 else "FAIL -- the two are NOT the same function; do not use for the A/B")


if __name__ == "__main__":
    main()
