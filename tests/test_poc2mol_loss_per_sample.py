"""The batched per-sample Poc2Mol loss must agree with the training criterion.

`max_poc2mol_loss` was calibrated against the old code path, which ran the criterion one
sample at a time inside `Poc2MolOutputDataset.__getitem__`. The batched replacement has to
reproduce those numbers exactly or the threshold silently changes meaning.

Run: venvPlixer/bin/python tests/test_poc2mol_loss_per_sample.py
"""
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch

from src.data.vox2smiles.poc2mol_inference import poc2mol_loss_per_sample
from src.models.pytorch3dunet_lib.unet3d.losses import get_loss_criterion

ALPHA, BETA = 1.0, 1.0
criterion = get_loss_criterion(
    {"name": "BCEDiceLoss", "weight": None, "normalization": "sigmoid",
     "alpha": ALPHA, "beta": BETA},
    with_logits=True,
)

torch.manual_seed(0)
B, C, S = 6, 9, 32
logits = torch.randn(B, C, S, S, S) * 3
target = (torch.rand(B, C, S, S, S) > 0.97).float()

batched = poc2mol_loss_per_sample(logits, target, alpha=ALPHA, beta=BETA)

reference = []
for i in range(B):
    parts = criterion(logits[i:i + 1], target[i:i + 1])
    reference.append(sum(parts.values()).item())
reference = torch.tensor(reference)

diff = (batched - reference).abs().max().item()
print(f"  per-sample losses (batched)  : {[round(v, 6) for v in batched.tolist()]}")
print(f"  per-sample losses (reference): {[round(v, 6) for v in reference.tolist()]}")
print(f"  max|diff| = {diff:.2e}")

ok = diff < 1e-5
print(f"  [{'PASS' if ok else 'FAIL'}] batched per-sample loss == criterion at batch size 1")

# A degenerate all-empty target must not produce NaN, since real ligands can have empty
# channels (iodine and bromine are empty in most complexes).
empty = poc2mol_loss_per_sample(logits, torch.zeros_like(target), alpha=ALPHA, beta=BETA)
finite = bool(torch.isfinite(empty).all())
print(f"  [{'PASS' if finite else 'FAIL'}] empty target gives finite loss: {empty[0].item():.4f}")

raise SystemExit(0 if (ok and finite) else 1)
