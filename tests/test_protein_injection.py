"""Semantics of protein-channel injection (experiment 1).

The baseline arm must be bit-identical to not having the feature, ligand-only data must
present as "protein absent", and masking must be train-only.

Run: venvPlixer/bin/python tests/test_protein_injection.py
"""
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch

from src.data.common.protein_channels import assemble_decoder_input, decoder_input_channels

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


B, L, P, S = 8, 9, 4, 8
ligand = torch.rand(B, L, S, S, S)
protein = torch.rand(B, P, S, S, S) + 0.5           # strictly positive, so zeroing is visible
has_protein = torch.tensor([True] * 5 + [False] * 3)  # 5 pocket samples, 3 ligand-only

print("\n1. baseline arm is untouched")
out = assemble_decoder_input(ligand, protein, has_protein, inject_protein=False)
check("returns the ligand tensor itself", out is ligand, f"shape {tuple(out.shape)}")

print("\n2. injected layout")
out = assemble_decoder_input(ligand, protein, has_protein, inject_protein=True)
check("channel count", out.shape[1] == decoder_input_channels(L, P, True) == 14,
      f"{out.shape[1]} channels")
check("ligand block preserved exactly", torch.equal(out[:, :L], ligand))

print("\n3. samples without protein read as absent")
zinc = ~has_protein
check("protein block zeroed for ligand-only samples",
      float(out[zinc, L:L + P].abs().max()) == 0.0)
check("flag plane is 0 for ligand-only samples",
      float(out[zinc, L + P].abs().max()) == 0.0)
check("protein block preserved for pocket samples",
      torch.equal(out[has_protein, L:L + P], protein[has_protein]))
check("flag plane is 1 for pocket samples",
      float(out[has_protein, L + P].min()) == 1.0)
check("flag plane is spatially constant",
      float(out[:, L + P].std(dim=(1, 2, 3)).max()) == 0.0)

print("\n4. masking")
gen = torch.Generator().manual_seed(0)
always = assemble_decoder_input(ligand, protein, has_protein, True,
                                mask_probability=1.0, training=True, generator=gen)
check("mask_probability=1.0 hides every protein",
      float(always[:, L:L + P].abs().max()) == 0.0 and float(always[:, L + P].abs().max()) == 0.0)
check("a masked pocket sample is indistinguishable from a ligand-only one",
      torch.equal(always[0, L:], always[6, L:]))

never = assemble_decoder_input(ligand, protein, has_protein, True,
                               mask_probability=1.0, training=False, generator=gen)
check("masking does not apply outside training",
      torch.equal(never[has_protein, L:L + P], protein[has_protein]))

counts = 0
trials = 400
for seed in range(trials):
    g = torch.Generator().manual_seed(seed)
    o = assemble_decoder_input(ligand, protein, has_protein, True,
                               mask_probability=0.25, training=True, generator=g)
    counts += int(o[has_protein, L + P].amax(dim=(1, 2, 3)).sum())
rate = 1 - counts / (trials * int(has_protein.sum()))
check("empirical mask rate matches mask_probability=0.25",
      abs(rate - 0.25) < 0.03, f"measured {rate:.3f}")

print()
if FAILURES:
    print(f"FAILED: {', '.join(FAILURES)}")
    raise SystemExit(1)
print("all checks passed")
