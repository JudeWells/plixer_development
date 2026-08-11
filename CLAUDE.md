# Plixer — generative Poc2Mol refactor

Branch `generative-poc2mol`. Everything measured before this branch is in **`report.md`**
(2,300 lines, §1–21) — consult it for detail; this file is the working brief.

Two-stage pocket-conditioned molecule generator: **Poc2Mol** (3D U-Net, protein voxels → ligand
voxels) → **Vox2Smiles** (ViT encoder + GPT-2 decoder, ligand voxels → SMILES).
Context: PhD thesis chapter. Paper: `plixer_ICML_GenBio_2025_version.pdf`.

---

## 1. The task, and why it is the right one

**Replace Poc2Mol's deterministic regression with a generative (flow-matching-like) model.**

Poc2Mol trains with BCE+Dice, so it predicts the **conditional mean** of ligand density given a
pocket. Every pathology measured on the predecessor branch follows from that one fact:

| measured | value | report.md |
|---|---|---|
| prediction vs truth, as an equivalent rigid rotation | **~27°** | §12g |
| every ligand channel over-emits | 1.33× (C) to 10.4× (I) | §18b |
| predicted carbon mass landing on real density | **58%** | §18b |
| at ≥60° the prediction overlaps a WRONG pose better than the truth does | 0.347 vs 0.334 | §12g |

That last row is the signature of a blurred field — what regression to the mean produces. The
density *is* genuinely pose-specific (true pose beats a 15° rotation in 85.6% of pockets, §12g),
so the information is present and the objective is destroying it.

**Encouraging counter-evidence (§18c):** the `other` channel is as sparse as chlorine (88.3%
empty) yet is the best-calibrated channel in the model (slope 0.90 vs 0.19). Where the pocket
genuinely determines the answer, the model commits. So this is an objective problem, not a
capacity or data problem.

**Yardstick for any new model:** `dice(pred, true)` against the rotation ladder in
`scripts/adhoc_analysis/poc2mol_rotation_control.py`. Beat **0.596** (≈ 27°).

---

## 2. Environment — two non-obvious gotchas

**`requirements.txt` IS the verified environment** (torch 2.3.1, lightning 2.3.2,
transformers 4.42.3, rdkit 2023.9.6). Don't trust `pip list` inside `venvPlixer`: that venv was
built by `uv` without its own `pip`, so bare `pip` resolves to a **conda base env** and reports
completely different versions. Introspect with the venv's own interpreter:

```bash
./venvPlixer/bin/python -c "import importlib.metadata as m; print(sorted((d.metadata['Name'],d.version) for d in m.distributions()))"
```

**Two fixes are required, not optional:**

1. **docktgrid bfloat16 patch** — `<venv>/lib/python3.11/site-packages/docktgrid/config.py` →
   `DTYPE = torch.bfloat16`. Every Poc2Mol config hardcodes `dtype: torch.bfloat16` while
   docktgrid ships float32. **Site-packages edit; does not survive a venv rebuild.** Intact on
   both nodes.
2. **`setuptools==80.9.0`** — 81+ removed `pkg_resources`, which `lightning_utilities` imports
   at module load. Not pinned in `requirements.txt`; a fresh install resolves 83 and crashes.

---

## 3. Data

Configs use paths relative to the repo root, so datasets sit **beside** the repo.

| path | contents |
|---|---|
| `../hiqbind/parquet_v2/{train,val,test}` | **use this** — float32 coords + per-atom features |
| `../zinc20_parquet_v2` | 5,537 files, 8,963,333 molecules, no `mol_block` |
| `../hiqbind/parquet`, `../zinc20_parquet` | v1, bfloat16-quantised coords. **Archival only** |
| `../hiqbind/raw_data_hiq_sm` | 36 GB raw structures (nebius2 only) |
| `../runs_n_poses/parquet/test/` | 1,836 unused eval systems (nebius2 only), report.md §6 |

**v2 schema:** `ligand_coords` (float32, (3,N) xyz-major), `ligand_coords_shape`,
`ligand_element_symbols`, `ligand_is_aromatic`, `ligand_n_hydrogens`, `ligand_is_acceptor`,
`ligand_formal_charge`, + HiQBind's protein columns and `system_id/smiles/split/cluster`.

⚠️ **Do not use v1.** Its coordinates are bfloat16-rounded — up to **0.49 Å error on a 0.75 Å
voxel**, on both model input and target (report.md §15c). v2 was regenerated to fix exactly
that, validated to bit-exactness against the source (§15f).

⚠️ **`../zinc20_parquet` (v1) must not be deleted** — raw ZINC20 mol2 is gone, so its
`mol_block` column is the only remaining source.

HiQBind is only **9,872 clusters**. At effective batch 1536 an epoch is ~6 optimiser steps —
convert every step-denominated schedule into epochs before trusting it (report.md §18f).

---

## 4. Code map — what to reuse

| component | path | note |
|---|---|---|
| **batched voxeliser** | `src/data/common/voxelization/batched.py` | fast (0.66 ms/sample), validated to 6e-7 against a float64 reference. Do not rewrite. |
| channel maps / view | `voxelization/{config,voxelizer}.py` | `LIGAND_CHANNELS_V2` (11ch); `UnifiedView` resolves feature tokens (`C_aromatic`, `N_withH`, `HBA`) |
| Poc2Mol + loss | `src/models/poc2mol.py`, `pytorch3dunet_lib/` | 3D U-Net; takes a time embedding with little change |
| provenance | `src/utils/provenance.py` | records git commit / config / parents into every ckpt. **Use it.** |
| rotation control | `scripts/adhoc_analysis/poc2mol_rotation_control.py` | the yardstick (§1) |
| emission audit | `scripts/adhoc_analysis/poc2mol_channel_emission.py` | over-emission, on-target, empty-channel mass |
| discrimination | `scripts/adhoc_analysis/poc2mol_scheme_discrimination.py` | scheme-agnostic; reads the channel map from the ckpt's own resolved config |
| past launchers | `scripts/launchers/` | recovered from /tmp; provenance for the experiments in report.md |

`tests/test_batched_voxelizer.py` and `test_eval_path_consistency.py` — run after touching
anything in `voxelization/`. ⚠️ **`pytest` is not installed in the venv.**

---

## 5. Metrics — which ones lie

🚨 **`val/loss` is unusable for Poc2Mol.** Three independent reasons:

1. **~76% of it is an irreducible floor.** A *perfect* prediction scores 0.6256:
   `compute_per_channel_dice` averages over all channels and only ~3.6 of 9 are occupied in a
   typical ligand, so an empty target channel contributes a full 1.0 (report.md §3c).
2. **The floor moves between channel schemes**, so 9ch and 11ch `val/loss` are not comparable
   at all — the 0.075 gap between them was entirely floor (§18a).
3. **It was ANTI-CORRELATED with downstream quality.** Over epochs 366→576 one run improved
   `val/loss` by 0.0069 while *losing* 0.0119 of discrimination — a within-run comparison, so no
   floor confound. Checkpointing on it actively selected a worse model (§18h).

**Use instead:** `val/emission/{ratio,on_target,empty_frac}` (logged in
`Poc2Mol.validation_step`, not floor-bound), the rotation-control Dice, and the composition
discrimination AUC. Read `on_target` *alongside* `empty_frac` — a model that simply emits less
everywhere lowers `empty_frac` without getting sharper.

**Dice's zero-gradient hole (§3c), which a generative objective should remove:** for an all-zero
target channel `intersect = (input*target).sum()` is identically 0, so `dice_coef ≡ 0` with
*exactly flat* gradient. Only BCE penalises that mass, and at `alpha: 1.0` it is ~56× smaller
than Dice. Result is a constant smear — fluorine emits 59 mass units when the ligand contains
**no fluorine**, and 75 when it does (§18b).

⚠️ Raising the BCE weight fixes calibration (ratios 1.9→1.1 at alpha 300) but **degrades Dice
monotonically** (+21.6% real error at alpha 300) and **buys nothing downstream** — a matched
stage-3 test found no effect (§19c). Keep `alpha: 1.0`. Don't repeat this experiment.

---

## 6. Pitfalls that have cost real time

**Hydra**
- A bare override fails for any key absent from the config node. Validate every override list
  with `python src/train.py --cfg job --resolve <overrides>` before launching anything long —
  Hydra reports only the *first* failure, so a clean dry run is the only proof.
- `{` is rejected in an override value, so checkpoint filename templates must live in YAML.
- `trainer/ddp.yaml` hardcodes `devices: 8`. `CUDA_VISIBLE_DEVICES` is **not** enough —
  Lightning errors rather than clamping. Pass `trainer.devices=N` per arm.

**Launching**
- Always health-check. A launcher that only backgrounded a process once reported 8 idle GPUs as
  "running" for hours after a composition error killed it in under a second. Every script in
  `scripts/launchers/` verifies PIDs are alive after 240–420 s and dumps the log tail on failure.

**Training config**
- `trainer.accumulate_grad_batches` in an experiment config is **ignored** — `src/train.py`
  overwrites it from `data.config.target_samples_per_batch / (batch_size × world_size)`. Set
  `target_samples_per_batch` explicitly; it is what keeps the effective batch matched across
  arms and world sizes.
- `logger=null_logger` is broken (`LearningRateMonitor` needs a logger). Use `WANDB_MODE=offline`.
- `init_weights_from` loads tensors only and starts at step 0; `ckpt_path` restores global_step
  + optimiser + scheduler. A new curriculum stage almost always wants the former.
- Weights-only checkpoints make a lossy resume (no Adam state). Keep `save_weights_only: False`.
- Size the LR decay to the **observed optimum**, not to `max_steps`. One run put decay at step
  6000 while early stopping fired at ~4500, so no arm ever saw the anneal; fixing it was worth
  +15% Tanimoto (§19d).

**Selection / evaluation**
- Validation must be deterministic for model selection (`rotate: false, translation: 0.0`,
  `use_cluster_member_zero: true`) — but that also makes augmentation-ensembling experiments a
  silent no-op, which wasted a full run (§20).
- "Best val X" from a stochastic-val run is inflated ~+0.025 by maximum-selection bias. Compare
  at matched validation counts or use a smoothed value.
- Ensemble members must be normalised with the **same** transform the metric applies (column
  z-norm). Row-standardising while the metric z-normalises by column inflated every ensemble
  figure by ~+0.005 (§20).

**Performance**
- Hoist `ProcessPoolExecutor` out of per-file loops — one pool per file re-imported torch+rdkit
  each time and turned a 20-minute job into a projected 10 days (§15f).
- CPU voxelisation is not an option: 2249 ms/sample vs 3.6 ms on GPU.

---

## 7. Nodes

| | host | role |
|---|---|---|
| **nebius1** | `computeinstance-e00k5jhwb42tm11726` | running the stage-1 ZINC v2 decoders (~60 h left, all 8 GPUs + all 128 cores) |
| **nebius2** | `ssh nebius2` (`computeinstance-e00xa3297a8mmtk4yp`) | 8× H100 idle — **use this for the generative work** |

Both: 8× H100 80 GB, 128 cores, 1.5 TB RAM, at `~/plixer_outer/plixer/`. The `plixer_outer/`
wrapper makes the configs' `../hiqbind/...` relative paths resolve.
Git remote is the **private** repo `git@github.com:JudeWells/plixer_development.git`.
W&B entity `cath`, projects `poc2mol` / `voxelSmiles`; credentials in `~/.netrc`.

⚠️ Anything run on nebius1 competes with the in-flight ZINC training for CPU — 2 arms × 4 ranks
× 14 dataloader workers already saturate the 128 cores.

**Best decoder checkpoint on disk** (mirrored on both nodes):
`checkpoints/best_decoder/s3_ab_baseline_step1250_auc0.7759.ckpt` — v1 pipeline, 9ch. The only
decoder that beats a parameter-free composition readout (0.7615), and only by +0.014. Treat
0.7759 as an optimistic point estimate; adjacent checks are 0.715/0.754 (report.md §21).

---

## 8. Worth knowing before you start

- **The decoder is not the binding constraint.** A 9-number composition readout taken off the
  density scores 0.7615 while the trained decoder scores ~0.72–0.78, and 24 swept decoder
  hyperparameters all landed within noise (§14d). The two readouts are near-**orthogonal**
  (r = +0.024), so their 50/50 blend reaches **0.785** — the best number in the project (§12d).
  Sharpening the density is therefore the high-value direction, which is what this branch is.
- **Protein channels into the decoder made things worse** and were dropped (§14a).
- **Model ensembling is worth +0.030**; augmentation ensembling adds only +0.002 on top of it,
  and subset selection over members is noise-fitting (§20).
- The paper's Table 1 and Table 2 come from **different checkpoints**, and the Vina numbers are
  irreproducible (results directory deleted). Regenerate both tables from a single checkpoint
  before quoting anything (report.md §4).
