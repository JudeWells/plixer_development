# Plixer — generative Poc2Mol refactor

Branch `generative-poc2mol`. Everything measured before this branch is in **`report.md`**
(2,456 lines, §1–22) — consult it for detail; this file is the working brief.
**§22 is the verdict on this branch: a negative result. Read §0 below first.**

Two-stage pocket-conditioned molecule generator: **Poc2Mol** (3D U-Net, protein voxels → ligand
voxels) → **Vox2Smiles** (ViT encoder + GPT-2 decoder, ligand voxels → SMILES).
Context: PhD thesis chapter. Paper: `plixer_ICML_GenBio_2025_version.pdf`.

---

## 0. STATUS: the generative branch is a NEGATIVE RESULT (concluded 2026-08-12)

**Flow-matching Poc2Mol does not beat regression Poc2Mol. Full write-up: report.md §22.**

| | regression | generative | |
|---|---|---|---|
| reconstruction Dice (1019 val pockets) | **0.5027** | 0.4350–0.47 | regression |
| stage-3 likelihood AUC znorm | **0.7522** | 0.7028 | regression |
| stage-3 Tanimoto | **0.1680** | 0.1474 | regression |
| confident false positives / pocket | 646 | **216** | generative |

Both stage-3 arms sit below the **parameter-free composition readout at 0.7615** (§14d).

**Why, in one line:** Dice and MSE are both minimised by the *conditional mean*, which is
what the regression model is trained to emit — so reconstruction metrics structurally
reward the blur this branch set out to remove. Evidence: Dice improves monotonically as the
ODE is integrated LESS (400 NFE → 0.254, 1 NFE → 0.355, no integration at all → 0.458), and
the `time_shift` sweep traces a continuum whose limit *is* the regression model.

**Do not re-derive these; read §1 and §5 with §22 in mind:**
- The yardstick is **0.5027** (n=1019), not 0.596 and not 0.4976 — see §1.
- Read a flow model out with `predict_expected(mode="one_step", k≈16, guidance≈3)`, never by
  sampling: worth +0.15 Dice at 1/100th the compute.
- `val/sample/dice` understates this model class by ~0.21; select on `val/mean/dice`.
- ZINC pretraining works (+0.01 Dice, 2× faster) and is worth keeping for any future variant.

**Top untried ideas** (report.md §22f has all eight): the **hybrid MSE+Dice loss**
(implemented as `loss_type: hybrid`, validated on the memorisation test, never trained at
scale) and **`voxel_aggregation: sum`** — the `max` aggregation destroys overlapping atoms
before either model sees the target (separability 0.07, §10b) and caps both equally.

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

⚠️ **2026-08-11: the motivating pathology does NOT reproduce on the current baseline.**
Head-to-head rotation control, 64 HiQBind v2 val pockets, same rotations for both models
(`poc2mol_v2_11ch_ep576` vs the flow arm at epoch 471, guidance 2.0):

| angle | regression dice | regression win | generative dice | generative win |
|---|---|---|---|---|
| true | **0.4976** | — | 0.2357 | — |
| 15 deg | 0.4599 | **0.938** | 0.2226 | 0.719 |
| 30 deg | 0.4076 | **0.977** | 0.2002 | 0.797 |
| 60 deg | 0.2998 | **0.992** | 0.1552 | 0.883 |
| 90 deg | 0.2576 | **1.000** | 0.1356 | 0.914 |
| 180 deg | 0.2790 | **0.984** | 0.1472 | 0.844 |

The regression model's win rate is 0.94-1.00 at EVERY angle here. The 12g result that
motivated this branch -- "at >=60 deg the prediction overlaps a WRONG pose better than the
truth" (0.347 vs 0.334, win rate < 0.5) -- was a DIFFERENT checkpoint on 104 PLINDER-panel
pockets and does not hold for the 11ch v2 model on HiQBind val. So the current baseline is
both more accurate AND more pose-specific than the generative model at this checkpoint.

That does not settle the question, but it does change what has to be argued. Dice(pred,true)
is maximised by the CONDITIONAL MEAN, so it structurally favours regression -- but the
rotation win rate was supposed to be the measurement immune to that, and regression wins it
too. The remaining case for the generative model is what Dice cannot express: multiple
draws per pocket (best-of-4 reaches 0.311 against 0.233 single-shot), diversity, and
downstream decoder metrics. **Decide this branch on decoder discrimination / Tanimoto, not
on reconstruction Dice.**

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

**Stage-1 11ch decoder — use the NEW file** (copied from nebius1, 2026-08-12 12:17 UTC):

| file | epoch / step | `val/loss` |
|---|---|---|
| **`checkpoints/s1_v2/s1_v2_11ch_ep14_step247256.ckpt`** | **14 / 247,256** | **0.0019** |
| `checkpoints/s1_v2/s1_v2_11ch.ckpt` (superseded) | 3 / 56,912 | 0.0136 |

Both are `last.ckpt` snapshots of the **same** nebius1 run `exp1_zinc_v2_11ch`
(`logs/exp1_zinc_v2_11ch/runs/2026-08-10_22-41-59`, wandb name `v2_11ch`, git 536299e), so the
`val/loss` comparison is within-run on the same monitor and is meaningful here — the §5 warning
is about Poc2Mol's Dice-floored loss, not the decoder's token cross-entropy. Full checkpoint
(optimiser + scheduler state present), 379 tensors, md5 `1a4f559ac0c9a6aad42a1803c996f587`.
**Point every new 11ch decoder init at the `_ep14_step247256` file.** Keep the old one only for
reproducing runs already launched against it.

⚠️ `configs/experiment/exp1_s3_v2_11ch.yaml` still has
`init_weights_from: checkpoints/s1_v2/s1_v2_11ch.ckpt` (the superseded epoch-3 file). Override
it on the command line or edit the config before launching anything new from it.

⚠️ That run is **still training on nebius1** — a fresher `last.ckpt` will exist later. Re-copy
rather than assuming epoch 14 is final.

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

---

## 9. The generative model — what exists now (2026-08-11)

Conditional **flow matching** (rectified flow), replacing BCE+Dice. `x_t = (1-t)x0 + t·x1`,
target velocity `x1 - x0`, MSE loss; a *sample* is a draw from p(ligand | pocket) rather than
its mean, so there is nothing left to average over poses.

| file | role |
|---|---|
| `src/models/flow_unet3d.py` | time-conditioned `ResidualUNetSE3D`. FiLM inside each residual block; encoder/decoder scaffolding duplicated so the vendored lib the regression model uses is untouched |
| `src/models/poc2mol_flow.py` | `Poc2MolFlow` — objective, ODE samplers (Euler/Heun), CFG, EMA, sample-based val metrics |
| `src/data/poc2mol/ligand_data_module.py` | ZINC ligand-only batches with an all-zero pocket, for stage A |
| `configs/model/poc2mol_flow.yaml` | every knob, with the reason for each default |
| `configs/data/poc2mol_flow_zinc_v2_11ch.yaml` | ZINC20 v2, 11ch, channel map identical to HiQBind's |
| `configs/experiment/flow_zinc_pretrain.yaml` | **stage A** — unconditional pretrain, 8 GPUs |
| `configs/experiment/flow_poc2mol_hiqbind.yaml` | **stage B** — pocket-conditioned, `init_weights_from` stage A |
| `scripts/adhoc_analysis/poc2mol_flow_eval.py` | sweeps guidance × steps × draws, then runs the rotation control. Imports the Dice from `poc2mol_rotation_control.py` so numbers stay comparable to **0.596** |
| `tests/test_flow_matching.py` | runnable directly (no pytest). Includes a memorisation test that catches sign errors in the path |

**Two-stage recipe.** HiQBind is 9,872 clusters ≈ 19 optimiser steps/epoch at batch 512 — thin
for a model that must learn what a ligand density *is* before learning which one a pocket
implies. Stage A trains that on ZINC20's 8.96M molecules with the pocket channels present and
identically zero, which is exactly the unconditional branch CFG needs; stage B loads it with
`init_weights_from` (not `ckpt_path`). Architectures are identical across stages by
construction — that is why `n_protein_channels: 4` is explicit in the ZINC data config.

**Selection metric is `val/sample/dice`, not `val/loss`.** The flow MSE says how well
velocities are predicted, not how good the samples are. `val/sample/dice` integrates the ODE
and scores with the pooled soft Dice of the rotation control. **Beat 0.4976** (see §1). Read it beside
`val/sample/emission_ratio` (1.0 = calibrated; the regression model runs 1.33–10.4× per
channel). Stage A is the exception: sample Dice is meaningless unconditionally, so it selects
on the (deterministic) flow loss.

**The Dice floor problem does not exist here** (§5, report.md §3c). MSE over velocity has no
per-channel averaging, and an empty target channel still carries gradient because its target
velocity `-x0` is non-zero noise — the exact hole that produced the constant fluorine smear.

**Knobs most worth sweeping**, in order: `guidance_scale` (sampling-time, free — sweep with
the eval script, no retraining), `time_shift` (1.0/2.0/3.0; higher-dimensional data wants more
of the path near noise), `time_sampling` (logit_normal vs uniform), `sample_steps`.

**Non-obvious things already paid for:**
- Validation runs on the **EMA** weights and `checkpoint["state_dict"]` **is** the EMA, so the
  metric a checkpoint is selected on is measured on the weights it contains. The EMA lives on
  the LightningModule, not a Callback: Lightning skips `_call_callbacks_on_save_checkpoint`
  entirely when `save_weights_only=True` — which every Poc2Mol experiment config sets — so a
  callback would have silently written online weights. Online weights ride along in
  `raw_state_dict` on full checkpoints.
- `save_on_train_epoch_end: False` is required, or the checkpoint is written before validation
  has produced the metric it is selected on.
- Per-timestep-bucket losses are logged **step-only and rank-zero-only**. A bucket can be empty
  for a given batch, and a metric some ranks log and others do not deadlocks DDP at epoch-level
  aggregation.
- The sinusoidal time features are computed in float32 and cast to the weight dtype at the
  projection. Consumers freeze Poc2Mol in bfloat16 (`Poc2MolInferenceBuilder._bind`) and a
  float32 input against bfloat16 weights is a hard error, not a silent upcast. Same reason
  `sample()` casts the ODE state at each network call while accumulating in float32.
- `Poc2MolInferenceBuilder` already branches on `model.is_generative`, so stage 3 can consume a
  flow checkpoint — but it costs `sample_steps` forward passes per batch instead of one. Budget
  for it, or cache predictions.

**The train/sample gap watchdog.** The known failure mode of a generative voxel model here
is that the training loss falls nicely and then *sampling* walks off the data manifold and
returns nonsense. The network only ever sees `x_t` ON the true path during training, so once
integration error pushes the state off it the velocities are extrapolations and the error
compounds. **`val/sample/dice` cannot detect this** — an untrained model and a diverged one
both score ~0. These can, and are logged every validation:

| metric | healthy | what it catches |
|---|---|---|
| `val/sample/traj_rms_max` | **~1** | every point of the probability path has RMS ~1 in model space, so this is ~1 when healthy *regardless of how much the model has learned*. Threshold-free alarm: >2 drifting, >10 diverging |
| `val/sample/rms_ratio` | ~1 | endpoint scale against the true data's own scale |
| `val/sample/out_of_range` | → 0 | voxels outside [-0.1, 1.1] occupancy **before** clamping. `to_occupancy` clamps, which would otherwise hide the failure completely (x=1e6 and x=1.5 give the same clamped grid) |
| `val/sample/max_abs_occ` | → 1 | the single worst voxel, pre-clamp |
| `val/sample/occupied_frac` | → `val/data/occupied_frac` (~0.003) | a diverged sample saturates everywhere |
| `val/restore/dice_t50` | > `val/sample/dice` | **assigns blame.** Sampling started from the TRUE path at t=0.5, same weights, same sampler |

Reading `val/restore/dice_t50` against `val/sample/dice`:

* **restore high, sample low** → the velocity field is fine, the *trajectory* is diverging.
  Remedies, in order: more `sample_steps`, `sampler: heun` (already default), lower
  `guidance_scale`, then `model.sample_clamp: 2.0`.
* **both low** → the model has not learned the field yet. The sampler is not the problem.
* `sample_clamp` is **off by default on purpose**: with it on, a diverging model still
  returns a plausible-looking saturated grid and the diagnostics understate the problem.

`tests/test_flow_matching.py` asserts the alarm actually fires — a deliberately diverged
model gives `traj_rms_max` 3.4e8 against 1.01 healthy, and the restoration probe scores 0.496
against 0.160 from noise on an under-trained model.

**Unconditional runs need the watchdog too.** `n_val_sample_pockets: 0` was originally set
for stage A on the grounds that an unconditional sample cannot be scored against a specific
held-out molecule -- but that switch gates the WHOLE sampling block, and the divergence
metrics need no pocket at all. Stage A is the run where the only other signal is the flow
loss falling (the reading that historically looked healthy while sampling was broken), and
its checkpoint seeds every stage-B arm. The config now sets `n_val_sample_pockets: 128` there.

Two of the metrics read differently without a pocket:

* `val/restore/dice_t50` is **fully meaningful** -- starting at t=0.5 puts the molecule's
  identity into `x_t`, so restoring to it is well-posed with or without conditioning.
* `val/sample/dice` is a **distribution check, not a quality score**: a perfect unconditional
  sample is simply a *different* molecule. Calibrate it against
  `dice(true_i, true_j)` for two different real molecules, which
  `scripts/adhoc_analysis/flow_sampling_watchdog.py` prints — **0.2779** on the ZINC val
  split. An unconditional model should approach that; a pocket-conditioned one must beat it.

**`scripts/adhoc_analysis/flow_sampling_watchdog.py`** asks the same questions of any
checkpoint without touching the run — point it at a live `last.ckpt` (rewritten every
validation) and optionally `--watch 900`. It reads the channel layout from the checkpoint, so
it handles both stages. Use it instead of restarting a healthy run to add metrics.

Stage A at step 10,000 (2026-08-11 14:36): `trajRMS 0.996`, `rmsRatio 0.994`, `oor 0.001`,
`occFrac 0.0021` against the data\'s own `0.0025`, **`restore_dice 0.928`**, `sample_dice
0.087` against the 0.278 target. Reading: no divergence anywhere, the velocity field near the
data is excellent, and the sampler already emits the right *amount* of density but not yet in
a molecule-like arrangement. A continuous watch is running to `/tmp/flow_zinc_watchdog.log`.

**`scripts/adhoc_analysis/flow_visualise_samples.py`** renders the true ligand beside several
independent draws, `--data hiqbind` drawing the pocket as a translucent shell so "did the
sample land in the cavity" is answerable by eye. Both loaders tolerate a live run rewriting
`last.ckpt` mid-read (`load_flow_checkpoint`), which is otherwise a 1.9 GB race that fails
with "PytorchStreamReader ... failed finding central directory".

First samples, 2026-08-11 15:20, in `evaluation_results/flow_samples/`:

* **ZINC, step 18k** — connected carbon skeletons with heteroatom decoration, occupied
  fraction 0.00228 against the data\'s 0.00219. Molecule-like, occasionally fragmented into
  two pieces. No divergence.
* **HiQBind conditional, step 3.5k, no pretraining** — samples land *inside* the pocket
  cavity and are molecule-shaped, but `on_target` is only 4-7% (regression Poc2Mol: 58% for
  carbon), soft-mass `emission_ratio` ~2.1 while thresholded `occupied_frac` is 0.0017
  against the data\'s 0.0032, and draws vary wildly — one produced 26 voxels. Early, but the
  failure mode is under-committing, not the regression model\'s constant smear.
* `val/restore/dice_t50` is **0.91** on the conditional model while `val/sample/dice` is
  0.15: the velocity field is already excellent near the data and it is the full trajectory
  from noise that is still weak. Textbook "keep training", not "the sampler is broken".

**Ligand size, ZINC vs HiQBind** (64 val ligands each, occupied voxels): means are close
(896 vs 989) but the spread is not — ZINC is tight (p10-p90 803-971, drug-like) where HiQBind
runs 492-1581 and beyond. Stage A therefore teaches a narrow size distribution; expect stage B
to spend capacity learning to produce both much smaller and much larger ligands.

**Running as of 2026-08-11 13:49 UTC on nebius2** (launched by
`scripts/launchers/run_flow_stage_a_and_baseline.sh`, logs under `/tmp/flow_launch_20260811_134919`):

| arm | GPUs | W&B | what | ETA |
|---|---|---|---|---|
| `flow_zinc_pretrain` | 2–7 | [x9jwxw0n](https://wandb.ai/cath/poc2mol/runs/x9jwxw0n) | stage A, 60k steps @ effective batch 384 (2.6 ZINC epochs) | ~4.5 h |
| `flow_hiqbind_scratch` | 0–1 | [qjvge1p8](https://wandb.ai/cath/poc2mol/runs/qjvge1p8) | the CONTROL: stage B settings with **no** pretraining, 1200 epochs @ effective batch 512 | ~10 h |

`scripts/launchers/run_flow_stage_b.sh` is waiting (nohup) and fires when stage A exits: it
picks stage A's best checkpoint by val/loss and starts three arms on GPUs 2–7 — `pre`
(identical to the control except `init_weights_from`, so the difference *is* the value of
pretraining), `shift3` (`time_shift=3.0`), `unif` (`time_sampling=uniform`). All four arms sit
at effective batch 512 so their `val/sample/dice` curves are comparable.

**Capacity is matched to the regression model**: 118.8M parameters against 117.2M. The extra
1.4% is the FiLM heads and the time MLP, so a win over 0.596 cannot be attributed to size.

⚠️ **Launch these ONLINE.** A first attempt ran with `WANDB_MODE=offline` (a habit from smoke
tests) and a live run cannot be flipped afterwards. Both launchers now set
`WANDB_MODE=online` and their health checks FAIL if no run URL appears within 300 s.

**Throughput: the pipeline is the constraint, not the batch size.** Measured 2026-08-11:
memory sits at 22% of 80 GB and mean SM utilisation is 41–79%, but **p10 SM is 0% and p90 is
100%** — the GPUs alternate between saturated and stalled, i.e. waiting on data. A pure
forward+backward at batch 64 runs at **≥680 samples/s/GPU** (measured on a *contended* card,
so a lower bound) against **242 samples/s/rank observed in training**: data-starved by ~3x,
matching §6\'s 300 vs 870. Raising the batch fills memory but cannot raise samples/s, and for
a fixed sample budget a larger effective batch means fewer optimiser steps, not more data.
The SMILES tokenisation the flow datamodule discards is only 1% of `__getitem__`, so that is
not the lever either; per-sample parquet extraction is (§6).

## Stage B as launched (2026-08-11 17:40)

Fine-tuning keeps **50% ZINC in the training mixture**. Two reasons: stage A bought a density
prior from 8.86M molecules that pocket-only fine-tuning would forget, and 9,872 clusters
overfit easily. The ZINC rows cost nothing conceptually — they are exactly the unconditional
branch CFG already trains.

**How the sources share a batch:** the ZINC dataset is configured `has_protein: true`, so
`StoredLigandComplex` (which reports `n_atoms_protein = 0`) emits the 4 protein slots
**empty**. Same 15-channel layout as a HiQBind record, no padding logic, and an all-zero
pocket is what the unconditional branch expects. Verified on a real batch: 34/30 split,
protein mass exactly **0.0** on ZINC rows and 662,559 on pocket rows.

**Per-source logging**, since an average cannot show ZINC being forgotten while the pocket
loss improves:

| | |
|---|---|
| train | `train/loss_pocket`, `train/loss_ligand_only`, `train/frac_pocket` (step-only, rank-zero — a batch can hold zero rows of a source) |
| val | two SEPARATE dataloaders → `val/hiqbind/*` and `val/zinc/*`, never a blend whose value moves with the mixture |

The pocket metrics are **mirrored to the unprefixed names** (`val/loss`, `val/sample/dice`),
so the checkpoint monitor and every comparison against the control arm keep working
unchanged. `add_dataloader_idx=False` throughout — letting Lightning append
`/dataloader_idx_N` would rename the metric checkpoints are selected on.

**Long warmup: 1500 steps (~40 epochs), lr 5e-5.** Stage A ends with its LR annealed to the
floor, so the incoming weights are a converged optimum of a *different* (unconditional)
objective; a short ramp would yank them away from it before the pocket channels mean
anything.

**The ablation ladder** — each arm differs from its neighbour by exactly one thing, all at
effective batch 512, all ~9,872 pocket samples per epoch, so they compare at matched EPOCHS:

| arm | GPUs | differs from previous by |
|---|---|---|
| `flow_hiqbind_scratch` (already running) | 0–1 | — |
| `flow_stage_b_pocket_only` | 2–3 | **pretraining** (stage-A weights) |
| `flow_stage_b_mixed` | 4–5 | **50% ZINC** in training |
| `flow_stage_b_mixed_shift3` | 6–7 | `time_shift=3.0` |

`pocket_only` carries the mixed arms' lr/warmup rather than its own config's, or "the value
of mixing" would confound the mixture with the schedule.

`scripts/launchers/run_flow_stage_b.sh` is nohup-waiting and fires automatically when stage A
exits. It refuses to launch if the stage-A checkpoint is missing or empty (rather than
silently training four from-scratch arms overnight) and its health check fails on a missing
W&B URL or a missing "state_dict loaded cleanly" line.

Then sweep guidance without retraining:
`poc2mol_flow_eval.py --ckpt <best> --guidance 1.0 1.5 2.0 3.0 --steps 25 50 --n_samples 4`.


---

## 10. Overnight 2026-08-11 22:30 -> 06:30 — the pipeline test

**Why the priority changed.** Reconstruction Dice cannot settle this branch: it is maximised
by the conditional mean, which is exactly what the regression Poc2Mol is trained to emit.
Measured head-to-head, the regression model also wins the rotation control on this split
(§1), so the generative case has to be made on what the pipeline is FOR — Tanimoto of
generated SMILES, and likelihood ranking of true binders.

**The readout discovery (no retraining).** A flow model can emit the conditional mean
directly: at t=0 the path velocity IS `x1 - x0`, so `x0 + v(x0, 0, c)` is a one-forward-pass
estimate of `E[ligand | pocket]`. Averaged over draws and with guidance, on 256 HiQBind val
pockets:

| readout | dice |
|---|---|
| single ODE sample, g=1 (what `val/sample/dice` logs) | 0.2081 |
| single ODE sample, g=3 | 0.2306 |
| `one_step` k=16, g=3 | 0.3635 |
| **`one_step` k=48, g=5** | **0.3966** |
| `one_step` k=48, g=8 / g=12 | 0.3922 / 0.3632 |
| regression Poc2Mol | 0.4976 |

Guidance optimum is ~5; more draws past k=16 buys little. This closes two thirds of the gap
for free, and `predict_expected` is what `Poc2MolInferenceBuilder` now uses to feed the
decoder (`generative_readout`, default `one_step`) — 2-4 forwards per batch instead of ~100
for a Heun trajectory, which is what makes a generative upstream affordable inside a
training loop at all.

**Running overnight** (all 8 GPUs):

| GPUs | arm | W&B |
|---|---|---|
| 0-1 | `s3_flow_11ch` — decoder on GENERATIVE density, one_step k=2 g=5 | [175hadv0](https://wandb.ai/cath/voxelSmiles/runs/175hadv0) |
| 4-5 | `s3_regression_11ch` — matched control, regression density | [p252tel6](https://wandb.ai/cath/voxelSmiles/runs/p252tel6) |
| 2-3 | `flow_stage_b_pocket_only_x2` — continuation from ep581, queued automatically | — |
| 6-7 | `flow_stage_b_mixed_shift033` — corrected timestep direction | [hp8lb9jq](https://wandb.ai/cath/poc2mol/runs/hp8lb9jq) |

Stopped: `flow_hiqbind_scratch` (flat at 0.211 for ~300 epochs, control complete) and
`flow_stage_b_mixed` (least distinguished of the flow arms). Both keep their W&B history.

The two s3 arms differ ONLY in which Poc2Mol supplies the density, so
`val/likelihood_auc_znorm` and `val/poc2mol/tanimoto` between them is the attributable
answer. `scripts/launchers/keep_gpus_busy.sh` runs the queue and, when `s3_flow_11ch`
finishes, fires `flow_multi_hypothesis_eval.py` — decode 8 independent voxel draws per
pocket and aggregate by max and mean, which is the capability the regression model does not
have. Status trail: `/tmp/overnight_queue.log`.

⚠️ **`pgrep -f "task_name=X"` matches the shell running it.** Two runs were killed this way.
Collect PIDs into a file first, or put the launch in a script file so the pattern is not in
the invoking command line.
