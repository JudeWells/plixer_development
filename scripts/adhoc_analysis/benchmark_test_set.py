"""Score one checkpoint over a large pocket x candidate panel, chunked, and save the matrix.

`fusion_ensemble.py` scores a pocket's whole candidate panel in a single forward pass
(`pixel_values[i:i+1].repeat(n_candidates, ...)`). At the 105-candidate PLINDER panel that is
fine; at the 943-candidate chronological panel it is a batch of 943 sequences through the ViT
encoder and the attention activations do not fit. This does the same arithmetic in chunks.

**The chunking must be numerically inert, and that is checked rather than asserted.** Run with
`--experiment <the 104-panel experiment>` and compare against `fusion_ensemble.py`'s
`single_decoder_mean`: they agree to ~1e-6 (bf16 autocast makes it non-bitwise, but the AUC is
identical). Do that before trusting any number from the large panel.

Also saves `system_ids` per row, which `fusion_ensemble.py` does not, because the whole point of
the large run is to slice subsets out of it afterwards:

    chronological  943 systems -- everything
    PLINDER        107         -- all inside the 943
    seq-sim        141         -- all inside the 943 (94 shared with PLINDER)

Subsets are extracted by taking ROWS of this matrix, so every report uses the same 943-candidate
panel. Re-scoring a subset against its own smaller panel would be an easier ranking task and the
numbers would not be comparable to each other or to the full split.

Usage (one checkpoint per GPU, then combine):
  CUDA_VISIBLE_DEVICES=0 ./venvPlixer/bin/python scripts/adhoc_analysis/benchmark_test_set.py \
      --experiment bench_chrono --checkpoint members/A1.ckpt --tag A1 \
      --out results/bench/A1.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import hydra                                                            # noqa: E402
from omegaconf import open_dict                                         # noqa: E402
from rdkit import Chem, RDLogger                                        # noqa: E402

from scripts.adhoc_analysis.poc2mol_scheme_discrimination import channel_counts  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="bench_chrono")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True, help="short name recorded in the output")
    p.add_argument("--out", required=True)
    p.add_argument("--n_aug", type=int, default=1,
                   help="augmentation replicates; 1 with --rotate off is the deterministic protocol")
    p.add_argument("--rotate", action="store_true")
    p.add_argument("--translation", type=float, default=6.0)
    p.add_argument("--cand_chunk", type=int, default=96,
                   help="candidates scored per forward. 96 keeps peak memory near the 105-panel "
                        "case that is known to fit alongside other jobs.")
    p.add_argument("--max_pockets", type=int, default=None, help="debug: stop early")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("PROJECT_ROOT", root)

    with hydra.initialize_config_dir(version_base="1.3",
                                     config_dir=os.path.join(root, "configs")):
        cfg = hydra.compose(config_name="train", overrides=[
            f"experiment={args.experiment}",
            "data.num_workers=4",
            "paths.output_dir=/tmp/bench", "paths.img_save_dir=/tmp/bench/img",
        ])
    os.makedirs("/tmp/bench/img", exist_ok=True)

    if args.rotate:
        with open_dict(cfg):
            cfg.data.val_datasets.roc_auc_plinder.complex_dataset.rotate = True
            cfg.data.val_datasets.roc_auc_plinder.complex_dataset.translation = args.translation
        print(f"augmentation ENABLED: rotate=True translation={args.translation}")
    else:
        print("augmentation DISABLED -- deterministic validation (the protocol the val metric uses)")

    channels = {int(k): list(v) for k, v in cfg.data.config.ligand_channels.items()}
    catch_all = bool(cfg.data.config.get("ligand_last_channel_is_catch_all", True))

    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    loader = datamodule.val_dataloader()[0]
    dataset = datamodule.val_datasets["roc_auc_plinder"]
    panel = list(dataset.decoy_smiles_list)

    # Row -> system_id. NOT available from the batch: the vox2smiles collate never forwards
    # system_id (the poc2mol dataset carries it as 'name'). It is recovered from the dataset's
    # own ordering instead, which is safe because the loader is shuffle=False and
    # use_cluster_member_zero pins one deterministic member per cluster. SMILES cannot be used
    # as the key -- the 943 systems carry only 916 distinct ligands.
    complex_dataset = dataset.complex_dataset
    ordered_system_ids = [complex_dataset.cluster_index[c][0]["system_id"]
                          for c in complex_dataset.cluster_ids]
    if len(ordered_system_ids) != len(complex_dataset):
        raise RuntimeError(
            f"recovered {len(ordered_system_ids)} system ids for {len(complex_dataset)} samples "
            "-- the row->system mapping would be wrong, and subset slicing depends on it")
    print(f"pockets: {len(ordered_system_ids)} ({len(set(ordered_system_ids))} unique)")

    counts = {}
    for smiles in panel:
        mol = Chem.MolFromSmiles(smiles)
        counts[smiles] = channel_counts(mol, channels, catch_all) if mol is not None else None
    valid_columns = np.array([counts[s] is not None for s in panel])
    n_channels = len(channels)
    candidate_counts = np.stack([counts[s] if counts[s] is not None else np.zeros(n_channels)
                                 for s in panel])
    print(f"panel: {len(panel)} candidates, {int(valid_columns.sum())} parse under RDKit")

    model = hydra.utils.instantiate(cfg.model).to(args.device).eval()
    tokenizer = model.tokenizer
    state = torch.load(args.checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(state.get("state_dict", state), strict=False)
    structural = [k for k in incompatible.missing_keys if k.startswith("model.")]
    if structural:
        raise RuntimeError(f"{args.checkpoint}: architecture mismatch ({structural[:4]})")
    del state

    decoder_all, mass_all, binder_all, sysid_all = [], [], [], []
    for a in range(args.n_aug):
        torch.manual_seed(4242 + a)
        np.random.seed(4242 + a)
        rows, mass_rows, binders = [], [], []
        started = time.time()
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                if args.max_pockets and len(rows) >= args.max_pockets:
                    break
                batch = {k: (v.to(args.device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                batch = datamodule.on_after_batch_transfer(batch)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pixel_values, info = model.build_pixel_values(batch, training=False)

                mass_rows.append(info["predicted"].float().sum(dim=(2, 3, 4)).cpu().numpy())

                cand_ids = batch["candidate_tokens"]["input_ids"].to(args.device)
                masked = cand_ids.clone()
                masked[masked == tokenizer.pad_token_id] = -100
                gather_idx = masked.clone()
                gather_idx[gather_idx == -100] = 0
                keep = masked != -100
                seq_len = keep.sum(dim=1).clamp(min=1)

                for i in range(pixel_values.size(0)):
                    if args.max_pockets and len(rows) >= args.max_pockets:
                        break
                    # THE ONLY DIFFERENCE FROM fusion_ensemble.py: the candidate panel is scored
                    # in chunks instead of one forward. Same tensors, same reduction, just
                    # assembled in pieces -- at 943 candidates the single-shot version OOMs.
                    parts = []
                    for start in range(0, cand_ids.size(0), args.cand_chunk):
                        stop = min(start + args.cand_chunk, cand_ids.size(0))
                        ids = cand_ids[start:stop]
                        repeated = pixel_values[i:i + 1].repeat(
                            ids.size(0), *([1] * (pixel_values.dim() - 1)))
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(repeated, labels=ids).logits
                        lp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                        tok = lp.gather(-1, gather_idx[start:stop].unsqueeze(-1)).squeeze(-1)
                        tok = tok * keep[start:stop]
                        parts.append((tok.sum(1) / seq_len[start:stop]).cpu().numpy())
                    rows.append(np.concatenate(parts))
                binders.extend(int(x) for x in batch["binder_indices"])

                if bi % 5 == 0:
                    done = len(rows)
                    rate = done / max(time.time() - started, 1e-9)
                    print(f"  aug{a} pocket {done}  ({rate*60:.1f}/min)", flush=True)

        decoder_all.append(np.stack(rows))
        mass_all.append(np.concatenate(mass_rows, axis=0)[:len(rows)])
        binder_all = binders[:len(rows)]
        sysid_all = ordered_system_ids[:len(rows)]
        print(f"  aug{a}: {decoder_all[-1].shape} in {(time.time()-started)/60:.1f} min", flush=True)

    positive = np.zeros((len(binder_all), len(panel)), dtype=bool)
    for row, binder in enumerate(binder_all):
        target = panel[binder] if binder < len(panel) else None
        if target is not None:
            positive[row] = np.array([s == target for s in panel])
        else:
            positive[row, binder] = True

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        decoder=np.stack(decoder_all),
        mass=np.stack(mass_all),
        candidate_counts=candidate_counts,
        positive=positive,
        valid_columns=valid_columns,
        panel=np.array(panel),
        system_ids=np.array([s if s is not None else "" for s in sysid_all]),
        checkpoint=np.array([args.checkpoint]),
        tag=np.array([args.tag]),
    )
    print(f"\nwrote {args.out}  decoder {np.stack(decoder_all).shape}  "
          f"{len(sysid_all)} system ids recorded")


if __name__ == "__main__":
    main()
