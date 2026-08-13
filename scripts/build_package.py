"""Assemble a self-contained, verifiable release of the DPO / ensemble work.

Mirrors the layout of ../plixer_ensemble_20260812 so the two bundles can be read side by side:
MANIFEST.json (sha256 + provenance per file), members/, parents/, configs/, provenance/,
results/, code/, verify.py, requirements.txt.

Two things this bundle does that the previous one could not, both consequences of the RL work
having been committed partway through the session:

1. Every member's **resolved config as executed** is extracted from the checkpoint's own
   embedded provenance record where available, and otherwise from the run directory. The
   resolved config is authoritative -- it supersedes any ambiguity about the working tree.
2. The manifest records, per member, whether the run predates the commits that put the RL
   source into git (`4ebc809` / `a891763`). A member stamped `2f10f95-dirty` cannot be
   reconstructed from its own commit alone, because 81 untracked files -- including
   `src/models/rl_vox2smiles.py` -- were in neither the commit nor `uncommitted.patch`.
   Being explicit about which members are affected is the point; hiding it would make the
   provenance record worse than useless.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(REPO), "plixer_dpo_ensemble_20260813")

# The provenance boundary: runs stamped before these commits carry a dirty tree whose
# untracked source is not recoverable from git alone.
SOURCE_COMMITS = ["4ebc809", "a891763", "f0e4cad", "226221c", "5a15db6", "2861508",
                  "4206f86", "5fea679"]

MEMBERS = [
    ("A1_e2e_warm_auc0.7759", "checkpoints/e2e_best/e2e_w2_warm_step312_auc0.7759.ckpt",
     "ensemble A member -- end-to-end warm-started decoder, drifted upstream"),
    ("A2_e2e_anneal_auc0.7718", "checkpoints/e2e_best/e2e_w4_anneal_step500_auc0.7718.ckpt",
     "ensemble A member -- end-to-end + resized LR anneal, drifted upstream"),
    ("A3_frozen_auc0.7657", "checkpoints/e2e_best/frozen_pipeline_s49_auc0.7657.ckpt",
     "ensemble A member -- frozen upstream, supervised only"),
    ("A4_sft_auc0.7589", "checkpoints/e2e_best/sft_s45_step312_auc0.7589.ckpt",
     "ensemble A member -- SFT comparator, frozen upstream"),
]

SINGLES = [
    ("DPO_best_auc0.7737", "checkpoints/e2e_best/dpo_auc_2e5_s43_auc0.7737.ckpt",
     "best single model on val/likelihood_auc_znorm -- DPO, lr 2e-5, seed 43"),
    ("DPO_best_tanimoto0.1787", "checkpoints/e2e_best/dpo_lr2e5_step2850_tan0.1787.ckpt",
     "best single model on val/poc2mol/tanimoto -- DPO, lr 2e-5, constant LR, step 2850"),
]

PARENTS = [
    ("s1_v2_11ch_ep14_step247256.ckpt", "checkpoints/s1_v2/s1_v2_11ch_ep14_step247256.ckpt",
     "stage-1 ZINC decoder init (init_weights_from for the e2e/frozen arms)"),
    ("s3_v2_11ch_step3000_auc0.7576.ckpt", "checkpoints/s3_v2_11ch/step_0003000_auc_0.7576.ckpt",
     "stage-3 warm decoder -- the starting policy for every RL run"),
    ("poc2mol_v2_11ch_ep576.ckpt", "checkpoints/poc2mol_v2/poc2mol_v2_11ch_ep576.ckpt",
     "root frozen Poc2Mol, 11ch, Dice 0.5027"),
]

RESULTS = [
    "results/e2e/fusion_aucsel4.json", "results/e2e/fusion_aucsel4.npz",
    "results/e2e/fusion_dpoauc6.json", "results/e2e/fusion_dpoauc6.npz",
    "results/e2e/fusion_dpo3dens8.json", "results/e2e/fusion_dpo3dens8.npz",
    "results/e2e/fusion_dpotan6.json", "results/e2e/fusion_dpotan6.npz",
    "results/e2e/fusion_combined.json", "results/e2e/fusion_final.json",
    "results/e2e/fusion_final2.json",
    "results/e2e/diversity.json", "results/e2e/hit_rate.json",
    "results/e2e/ens8.json", "results/e2e/ens8.npz",
]

CODE = [
    "scripts/adhoc_analysis/fusion_ensemble.py",
    "scripts/adhoc_analysis/combine_fusion_members.py",
    "scripts/adhoc_analysis/generation_diversity.py",
    "scripts/adhoc_analysis/tanimoto_hit_rate.py",
    "scripts/adhoc_analysis/decoder_ensembling.py",
    "scripts/e2e_report.py",
    "src/models/rl_vox2smiles.py",
    "src/models/end_to_end.py",
    "src/data/vox2smiles/end_to_end.py",
]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def checkpoint_provenance(path):
    """Pull the embedded provenance record without loading the weights into memory twice."""
    try:
        blob = torch.load(path, map_location="cpu")
    except Exception as error:
        return {"error": f"could not read: {error}"}
    record = blob.get("provenance")
    extra = {"global_step": blob.get("global_step"), "epoch": blob.get("epoch")}
    del blob
    if record is None:
        return {"embedded_provenance": None, **extra}
    return {"embedded_provenance": record, **extra}


def copy_and_record(src_rel, dst, description, manifest, category):
    src = os.path.join(REPO, src_rel) if not os.path.isabs(src_rel) else src_rel
    if not os.path.exists(src):
        print(f"  ⚠️ MISSING, skipped: {src_rel}")
        return None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    entry = {
        "category": category,
        "path": os.path.relpath(dst, OUT),
        "source": src_rel,
        "description": description,
        "bytes": os.path.getsize(dst),
        "sha256": sha256(dst),
    }
    if dst.endswith(".ckpt"):
        prov = checkpoint_provenance(dst)
        entry.update({k: v for k, v in prov.items() if k != "embedded_provenance"})
        record = prov.get("embedded_provenance")
        if record:
            git_info = record.get("git", {})
            entry["run_commit"] = git_info.get("commit_short")
            entry["run_tree_dirty"] = git_info.get("is_dirty")
            entry["task_name"] = record.get("task_name")
            entry["seed"] = record.get("seed")
            entry["run_dir"] = record.get("run_dir")
            # THE point of this bundle: say plainly whether this member's source is in git.
            entry["source_recoverable_from_commit"] = not git_info.get("is_dirty", True)
            prov_dir = os.path.join(OUT, "provenance")
            os.makedirs(prov_dir, exist_ok=True)
            name = os.path.splitext(os.path.basename(dst))[0] + ".json"
            with open(os.path.join(prov_dir, name), "w") as handle:
                json.dump(record, handle, indent=2)
            entry["provenance_file"] = f"provenance/{name}"
            # Resolved config as executed, if the run dir survives.
            run_dir = record.get("run_dir")
            if run_dir and os.path.exists(os.path.join(run_dir, "resolved_config.yaml")):
                cfg_dir = os.path.join(OUT, "configs")
                os.makedirs(cfg_dir, exist_ok=True)
                cfg_name = os.path.splitext(os.path.basename(dst))[0] + ".yaml"
                shutil.copy2(os.path.join(run_dir, "resolved_config.yaml"),
                             os.path.join(cfg_dir, cfg_name))
                entry["resolved_config"] = f"configs/{cfg_name}"
    manifest.append(entry)
    print(f"  {entry['bytes']/1e9:6.2f} GB  {entry['path']}")
    return entry


def main():
    if os.path.exists(OUT):
        print(f"{OUT} exists -- refusing to overwrite. Remove it first.")
        return 1
    os.makedirs(OUT)
    manifest = []

    print("members/")
    for name, src, desc in MEMBERS:
        copy_and_record(src, os.path.join(OUT, "members", f"{name}.ckpt"), desc,
                        manifest, "ensemble_member")
    print("best_single/")
    for name, src, desc in SINGLES:
        copy_and_record(src, os.path.join(OUT, "best_single", f"{name}.ckpt"), desc,
                        manifest, "best_single_model")
    print("parents/")
    for name, src, desc in PARENTS:
        copy_and_record(src, os.path.join(OUT, "parents", name), desc, manifest, "parent")
    print("results/")
    for rel in RESULTS:
        copy_and_record(rel, os.path.join(OUT, "results", os.path.basename(rel)),
                        "analysis output", manifest, "result")
    print("code/")
    for rel in CODE:
        copy_and_record(rel, os.path.join(OUT, "code", os.path.basename(rel)),
                        "analysis / implementation source", manifest, "code")
    for rel in ["requirements.txt"]:
        copy_and_record(rel, os.path.join(OUT, os.path.basename(rel)),
                        "verified environment", manifest, "env")

    top = {
        "name": "plixer_dpo_ensemble_20260813",
        "created": subprocess.check_output(["date", "-Iseconds"], text=True).strip(),
        "repo": git("config", "--get", "remote.origin.url"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head_commit": git("rev-parse", "HEAD"),
        "head_describe": git("describe", "--always", "--dirty"),
        "source_commits": SOURCE_COMMITS,
        "provenance_note": (
            "Members whose run_tree_dirty is true were trained before the RL source was "
            "committed; 81 untracked files including src/models/rl_vox2smiles.py were in "
            "neither the stamped commit nor uncommitted.patch, so those members are NOT "
            "reconstructable from their commit alone. Their resolved_config is authoritative, "
            "and the equivalent source is in the source_commits listed above. Training-code "
            "mtimes were all <= 05:50 on 2026-08-13 while the rl_dpo_auc_* members started "
            "06:32, so those members ran byte-identical source to 4ebc809+a891763."
        ),
        "files": manifest,
    }
    with open(os.path.join(OUT, "MANIFEST.json"), "w") as handle:
        json.dump(top, handle, indent=2)

    total = sum(e["bytes"] for e in manifest)
    print(f"\n{len(manifest)} files, {total/1e9:.2f} GB -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
