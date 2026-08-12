"""Package the 0.7883 fusion ensemble into a self-contained, reproducible release bundle.

The six members live under `logs/<task>/runs/<timestamp>/checkpoints/`, which is a working
directory, not an artefact: it is bound to one machine, it is pruned by `save_top_k`, and it
disappears when the node does. This copies everything needed to re-derive the reported number
on a different machine into one directory, and records enough provenance that each file can be
traced back to the code and data that produced it.

What goes in, and why each piece is needed rather than nice to have:

    members/     the six end-to-end checkpoints that ARE the ensemble. Each carries both
                 models (`poc2mol.*` and `model.*`), so a member is self-sufficient at
                 inference and its decoder cannot be accidentally paired with the wrong
                 upstream.
    upstreams/   the three distinct frozen Poc2Mol densities the members were trained against.
                 Redundant with members/ by construction, included because the DIFFERENCE
                 between them is the experimental variable and a reader will want them alone.
    parents/     the two root checkpoints every member descends from -- the stage-1 decoder
                 initialisation and the original Poc2Mol. Without these the chain cannot be
                 re-run from scratch, only re-evaluated.
    configs/     the fully resolved Hydra config per member, as actually executed.
    provenance/  git commit, branch, dirty state, package versions, CUDA, argv, parents.
    results/     the fusion evaluation output and the cached member score matrices, so the
                 blend curve can be recomputed without a GPU.
    code/        the two scripts needed to reproduce the evaluation and the extraction.

Every file is sha256'd into MANIFEST.json. Hashing ~16 GB takes a few minutes and is the
difference between "these are the weights" and "we believe these are the weights".

Usage:
    ./venvPlixer/bin/python scripts/package_ensemble_release.py --out release/plixer_ensemble_20260812
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The six ensemble members, as (release name, arm, upstream key, source path).
# Names encode arm + seed + step + the val AUC the checkpoint was selected on, so a bare file
# in a downloads folder is still self-describing.
MEMBERS = [
    ("Z_seed42_step0750_auc0.7617", "Z", "original",
     "logs/e2e_z_frozen_r3/runs/2026-08-12_17-28-31/checkpoints/step_0000750_auc_0.7617.ckpt"),
    ("Z_seed43_step0750_auc0.7721", "Z", "original",
     "logs/e2e_z_frozen_s43/runs/2026-08-12_18-26-35/checkpoints/step_0000750_auc_0.7721.ckpt"),
    ("I_seed42_step0750_auc0.7706", "I", "e2e_step500",
     "logs/e2e_i_frozen_on_step500_density_r4/runs/2026-08-12_20-03-08/checkpoints/step_0000750_auc_0.7706.ckpt"),
    ("I_seed43_step1250_auc0.7713", "I", "e2e_step500",
     "logs/e2e_i_frozen_on_step500_density_s43/runs/2026-08-12_20-32-14/checkpoints/step_0001250_auc_0.7713.ckpt"),
    ("H_seed42_step1250_auc0.7689", "H", "e2e_step1000",
     "logs/e2e_h_frozen_on_early_density_r4/runs/2026-08-12_19-29-07/checkpoints/step_0001250_auc_0.7689.ckpt"),
    ("H_seed43_step1250_auc0.7714", "H", "e2e_step1000",
     "logs/e2e_h_frozen_on_early_density_s43/runs/2026-08-12_20-42-13/checkpoints/step_0001250_auc_0.7714.ckpt"),
]

UPSTREAMS = {
    "original": ("checkpoints/poc2mol_v2/poc2mol_v2_11ch_ep576.ckpt",
                 "Poc2Mol v2, 11 channel, epoch 576. BCE+Dice only, never trained end-to-end."),
    "e2e_step500": ("checkpoints/e2e_b_r3_poc2mol_step500.ckpt",
                    "The above, plus 500 optimiser steps of END-TO-END training in arm B "
                    "(LM cross-entropy + BCE+Dice, upstream lr 1e-4), then frozen."),
    "e2e_step1000": ("checkpoints/e2e_b_r3_poc2mol_step1000.ckpt",
                     "The above, plus 1000 optimiser steps of the same end-to-end training, "
                     "then frozen."),
}

PARENTS = [
    ("checkpoints/s1_v2/s1_v2_11ch_ep14_step247256.ckpt",
     "Stage-1 decoder (ZINC, ligand-only, 11 channel). EVERY member's decoder starts here."),
    ("checkpoints/poc2mol_v2/poc2mol_v2_11ch_ep576.ckpt",
     "Root Poc2Mol. Both end-to-end upstreams are this checkpoint trained further."),
]

RESULTS = [("/tmp/fusion_full.json", "fusion_full.json"),
           ("/tmp/fusion_members.npz", "fusion_members.npz")]

CODE = ["scripts/adhoc_analysis/fusion_ensemble.py",
        "scripts/adhoc_analysis/extract_poc2mol_from_e2e.py",
        "scripts/adhoc_analysis/e2e_sweep_report.py"]


def sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return {"bytes": os.path.getsize(dst), "sha256": sha256(dst)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip_hash", action="store_true",
                        help="skip sha256 (fast dry run; the manifest is then NOT verifiable)")
    args = parser.parse_args()

    global sha256
    if args.skip_hash:
        sha256 = lambda p, chunk=None: "SKIPPED"  # noqa: E731

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    print(f"packaging -> {out}")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    manifest = {
        "bundle": os.path.basename(out),
        "created_utc": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                      capture_output=True, text=True).stdout.strip(),
        "git_commit_at_packaging": commit,
        "headline": {
            "metric": "val/likelihood_auc_znorm",
            "panel": "PLINDER 104 pockets x 105 candidates (hiqbind parquet_v2 test split)",
            "fusion_ensemble_auc": 0.7883,
            "blend_weight_composition": 0.2,
            "decoder_ensemble_only_auc": 0.7818,
            "single_deterministic_checkpoint_auc": 0.7612,
        },
        "members": [], "upstreams": [], "parents": [], "results": [], "code": [],
    }

    print("\nmembers:")
    for name, arm, upstream, src in MEMBERS:
        source = os.path.join(ROOT, src)
        if not os.path.exists(source):
            print(f"  MISSING {src}"); continue
        info = copy(source, os.path.join(out, "members", name + ".ckpt"))

        run_dir = os.path.dirname(os.path.dirname(source))
        for fname, subdir in (("resolved_config.yaml", "configs"),
                              ("provenance.json", "provenance")):
            candidate = os.path.join(run_dir, fname)
            if os.path.exists(candidate):
                ext = os.path.splitext(fname)[1]
                copy(candidate, os.path.join(out, subdir, name + ext))

        checkpoint = torch.load(source, map_location="cpu")
        embedded = checkpoint.get("provenance", {}) or {}
        entry = {
            "name": name, "arm": arm, "upstream": upstream,
            "file": f"members/{name}.ckpt", **info,
            "global_step": checkpoint.get("global_step"),
            "epoch": checkpoint.get("epoch"),
            "seed": embedded.get("seed"),
            "task_name": embedded.get("task_name"),
            "git_commit": (embedded.get("git") or {}).get("commit"),
            "git_branch": (embedded.get("git") or {}).get("branch"),
            "git_dirty": (embedded.get("git") or {}).get("is_dirty"),
            "git_untracked": (embedded.get("git") or {}).get("untracked_files"),
            "argv": (embedded.get("env") or {}).get("argv"),
            "packages": (embedded.get("env") or {}).get("packages"),
            "parent_checkpoints": embedded.get("parent_checkpoints"),
            "source_path_on_node": src,
        }
        manifest["members"].append(entry)
        print(f"  {name}  {info['bytes']/1e9:.2f} GB  step {checkpoint.get('global_step')}  "
              f"seed {embedded.get('seed')}  {str(info['sha256'])[:16]}")
        del checkpoint

    print("\nupstreams:")
    for key, (src, description) in UPSTREAMS.items():
        source = os.path.join(ROOT, src)
        if not os.path.exists(source):
            print(f"  MISSING {src}"); continue
        info = copy(source, os.path.join(out, "upstreams", os.path.basename(src)))
        manifest["upstreams"].append({"key": key, "file": f"upstreams/{os.path.basename(src)}",
                                      "description": description, **info,
                                      "source_path_on_node": src})
        print(f"  {key:<14} {os.path.basename(src)}  {info['bytes']/1e9:.2f} GB")

    print("\nparents:")
    seen = set()
    for src, description in PARENTS:
        if src in seen:
            continue
        seen.add(src)
        source = os.path.join(ROOT, src)
        if not os.path.exists(source):
            print(f"  MISSING {src}"); continue
        info = copy(source, os.path.join(out, "parents", os.path.basename(src)))
        manifest["parents"].append({"file": f"parents/{os.path.basename(src)}",
                                    "description": description, **info,
                                    "source_path_on_node": src})
        print(f"  {os.path.basename(src)}  {info['bytes']/1e9:.2f} GB")

    print("\nresults + code:")
    for src, name in RESULTS:
        if os.path.exists(src):
            info = copy(src, os.path.join(out, "results", name))
            manifest["results"].append({"file": f"results/{name}", **info})
            print(f"  results/{name}")
    for src in CODE:
        source = os.path.join(ROOT, src)
        if os.path.exists(source):
            info = copy(source, os.path.join(out, "code", os.path.basename(src)))
            manifest["code"].append({"file": f"code/{os.path.basename(src)}",
                                     "repo_path": src, **info})
            print(f"  code/{os.path.basename(src)}")

    for extra in ("requirements.txt",):
        source = os.path.join(ROOT, extra)
        if os.path.exists(source):
            copy(source, os.path.join(out, extra))
            print(f"  {extra}")

    with open(os.path.join(out, "MANIFEST.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)

    total = sum(e["bytes"] for group in ("members", "upstreams", "parents", "results", "code")
                for e in manifest[group])
    print(f"\nwrote MANIFEST.json — {len(manifest['members'])} members, {total/1e9:.1f} GB total")


if __name__ == "__main__":
    main()
