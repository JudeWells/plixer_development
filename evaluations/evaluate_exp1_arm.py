"""Evaluate one experiment-1 decoder arm on the HiQBind test pockets.

Produces the numbers the baseline-vs-protein-channel comparison is decided on, from a single
vox2smiles checkpoint plus the shared frozen Poc2Mol. Emits JSON so arms and seeds can be
aggregated by `compare_exp1_arms.py`.

Metrics, and why each is here:

``teacher_forced_loss`` / ``token_accuracy``
    The training objective. Cheap and low-variance, but it measures next-token prediction,
    not molecule quality -- never decide on this alone.

``validity`` / ``uniqueness``
    Sanity floors. A model can win on loss while emitting unparseable SMILES.

``tanimoto_to_true``
    Morgan/Tanimoto between the generated molecule and the pocket's real ligand. The direct
    "did it find the right chemistry" measure.

``likelihood_auc_raw`` / ``likelihood_auc_znorm``
    Rank the true ligand against decoys by mean per-token log-likelihood. **The raw number
    is dominated by a ligand-size nuisance term** -- CLAUDE.md §5.1 measured the ligand
    effect at 84% of variance against 9.5% for the pocket-ligand interaction, and a
    pocket-blind baseline scores exactly 0.500. The z-normalised variant removes it by
    standardising each ligand's scores across pockets, and is the number to quote.

``--mask-protein`` runs the protein arm with its protein channels zeroed and the flag off.
If that recovers baseline performance, the model is genuinely using the pocket; if it does
not degrade, the protein channels are being ignored and any win came from elsewhere.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.common.protein_channels import assemble_decoder_input
from src.data.common.tokenizers.smiles_tokenizer import build_smiles_tokenizer
from src.data.common.voxelization.config import Poc2MolDataConfig
from src.data.poc2mol.collate import VoxelBatchBuilder, collate_complex_records
from src.data.poc2mol.datasets import ParquetDataset
from src.data.vox2smiles.poc2mol_inference import poc2mol_loss_per_sample
from src.models.poc2mol import Poc2Mol, ResUnetConfig
from src.models.vox2smiles import VoxToSmilesModel

RDLogger.DisableLog("rdApp.*")

POC2MOL_LOSS = {"name": "BCEDiceLoss", "weight": None, "normalization": "sigmoid",
                "alpha": 1.0, "beta": 1.0}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vox2smiles_ckpt", required=True)
    p.add_argument("--poc2mol_ckpt",
                   default="checkpoints/exp1_shared_poc2mol/poc2mol_cons4_drop_epoch307.ckpt")
    p.add_argument("--data_path", default="../hiqbind/parquet/test")
    p.add_argument("--inject_protein", action="store_true",
                   help="protein arm: decoder input is [ligand | protein | flag] = 14 channels")
    p.add_argument("--mask_protein", action="store_true",
                   help="ablation: zero the protein channels and clear the flag")
    p.add_argument("--n_pockets", type=int, default=0, help="0 = all")
    p.add_argument("--panel_size", type=int, default=0,
                   help="decoys to rank against; 0 = every test ligand. The likelihood "
                        "matrix costs pockets x panel decoder passes, so cap it for a quick look.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_models(args, device):
    poc2mol = Poc2Mol(
        config=ResUnetConfig(in_channels=4, out_channels=9,
                             layer_order="gcrd", dropout_prob=0.1),
        loss=POC2MOL_LOSS,
    )
    ck = torch.load(args.poc2mol_ckpt, map_location="cpu")
    poc2mol.model.load_state_dict({k.replace("model.", ""): v for k, v in ck["state_dict"].items()})
    poc2mol = poc2mol.to(torch.bfloat16).to(device).eval()

    dec_ck = torch.load(args.vox2smiles_ckpt, map_location="cpu")
    hparams = dec_ck.get("hyper_parameters", {})
    decoder = VoxToSmilesModel(**hparams) if hparams else None
    if decoder is None:
        raise RuntimeError(
            f"{args.vox2smiles_ckpt} carries no hyper_parameters; cannot rebuild the decoder. "
            "Checkpoints written by src/train.py always do."
        )
    decoder.load_state_dict(dec_ck["state_dict"])
    decoder = decoder.to(device).eval()

    expected = 14 if args.inject_protein else 9
    actual = decoder.hparams.config.num_channels
    if actual != expected:
        raise ValueError(
            f"--inject_protein={args.inject_protein} implies {expected} input channels but the "
            f"checkpoint was built with {actual}. The arms' weights are not interchangeable."
        )
    return poc2mol, decoder


def morgan(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    poc2mol, decoder = load_models(args, device)
    tokenizer = build_smiles_tokenizer()

    cfg = Poc2MolDataConfig(batch_size=args.batch_size,
                            random_rotation=False, random_translation=0.0)
    ds = ParquetDataset(config=cfg, data_path=args.data_path, rotate=False,
                        translation=0.0, use_cluster_member_zero=True)
    if args.n_pockets:
        ds.cluster_ids = ds.cluster_ids[: args.n_pockets]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_complex_records)
    builder = VoxelBatchBuilder(cfg, n_protein_channels=4)

    losses, accs, generated, truths, poc_losses = [], [], [], [], []
    decoder_inputs = []

    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        grids = builder(batch)
        protein, true_ligand = grids["protein"], grids["ligand"]

        logits = poc2mol.model(x=protein)
        predicted = torch.sigmoid(logits)
        poc_losses.append(poc2mol_loss_per_sample(logits, true_ligand).float().cpu())

        has_protein = torch.ones(len(predicted), dtype=torch.bool, device=device)
        if args.mask_protein:
            has_protein = torch.zeros_like(has_protein)
        pixel_values = assemble_decoder_input(
            predicted, protein, has_protein, args.inject_protein,
            mask_probability=0.0, training=False,
        )

        smiles = list(grids["smiles"])
        tok = tokenizer([tokenizer.bos_token + s + tokenizer.eos_token for s in smiles],
                        padding="max_length", max_length=200, truncation=True,
                        return_tensors="pt")
        labels = tok["input_ids"].to(device)

        out = decoder(pixel_values, labels=labels)
        losses.append(float(out.loss))
        masked = labels.clone()
        masked[masked == tokenizer.pad_token_id] = -100
        from src.utils.metrics import accuracy_from_outputs
        accs.append(float(accuracy_from_outputs(out, masked, start_ix=1, ignore_index=-100)))

        generated.extend(decoder.generate_smiles(pixel_values, max_attempts=1))
        truths.extend(smiles)
        decoder_inputs.append(pixel_values.cpu())

    # ---- generation quality
    valid = [s for s in generated if s and Chem.MolFromSmiles(s) is not None]
    tan = []
    for gen, true in zip(generated, truths):
        fg, ft = morgan(gen), morgan(true)
        if fg is not None and ft is not None:
            tan.append(DataStructs.TanimotoSimilarity(fg, ft))

    # ---- likelihood ranking: pocket x candidate-ligand matrix
    #
    # The decoy panel is the test set's own ligands, matching the convention in
    # evaluate_combined_vox2smiles.py so the numbers stay comparable with the published
    # tables. Positives are found by SMILES identity rather than row index: 27 test systems
    # share a SMILES with another (CLAUDE.md §5.1), so a pocket can have several correct
    # answers in the panel and index-based labelling would score those as misses.
    panel = truths if not args.panel_size else truths[: args.panel_size]
    tok_panel = tokenizer([tokenizer.bos_token + s_ + tokenizer.eos_token for s_ in panel],
                          padding="max_length", max_length=200, truncation=True,
                          return_tensors="pt")["input_ids"]

    panel_index = defaultdict(list)
    for j, smi in enumerate(panel):
        panel_index[smi].append(j)

    matrix = np.full((len(truths), len(panel)), np.nan, dtype=np.float64)
    row = 0
    for chunk in decoder_inputs:
        for i in range(len(chunk)):
            vox = chunk[i : i + 1].to(device)
            scores = []
            for start in range(0, len(panel), 64):
                ids = tok_panel[start : start + 64].to(device)
                out_ = decoder(vox.repeat(len(ids), 1, 1, 1, 1), labels=ids)
                lp = torch.log_softmax(out_.logits.float(), dim=-1)
                m = ids.clone()
                m[m == tokenizer.pad_token_id] = -100
                g = m.clone()
                g[g == -100] = 0
                tlp = lp.gather(-1, g.unsqueeze(-1)).squeeze(-1) * (m != -100)
                scores.append((tlp.sum(1) / (m != -100).sum(1).clamp(min=1)).cpu().numpy())
            matrix[row] = np.concatenate(scores)
            row += 1

    # z-normalise each CANDIDATE's column across pockets. Raw mean log-likelihood is
    # dominated by intrinsic molecule likelihood (mostly size): 84% of variance is the
    # ligand effect vs 9.5% for the pocket-ligand interaction, and a pocket-blind baseline
    # scores exactly 0.500. Standardising per column removes that nuisance term.
    z = (matrix - np.nanmean(matrix, axis=0)) / (np.nanstd(matrix, axis=0) + 1e-9)

    raw, znorm = [], []
    for i, true_smi in enumerate(truths):
        positives = panel_index.get(true_smi)
        if not positives or len(positives) == len(panel):
            continue                      # true ligand absent from the panel, or all of it
        labels_vec = np.zeros(len(panel), dtype=int)
        labels_vec[positives] = 1
        raw.append(roc_auc_score(labels_vec, matrix[i]))
        znorm.append(roc_auc_score(labels_vec, z[i]))

    result = {
        "vox2smiles_ckpt": args.vox2smiles_ckpt,
        "poc2mol_ckpt": args.poc2mol_ckpt,
        "inject_protein": args.inject_protein,
        "mask_protein": args.mask_protein,
        "n_pockets": int(matrix.shape[0]),
        "panel_size": int(matrix.shape[1]),
        "n_pockets_scored": len(raw),
        "teacher_forced_loss": float(np.mean(losses)),
        "token_accuracy": float(np.mean(accs)),
        "poc2mol_loss": float(torch.cat(poc_losses).mean()),
        "validity": len(valid) / max(len(generated), 1),
        "uniqueness": len(set(valid)) / max(len(valid), 1),
        "tanimoto_to_true_mean": float(np.mean(tan)) if tan else None,
        "tanimoto_to_true_median": float(np.median(tan)) if tan else None,
        "likelihood_auc_raw": float(np.mean(raw)) if raw else None,
        "likelihood_auc_znorm": float(np.mean(znorm)) if znorm else None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
