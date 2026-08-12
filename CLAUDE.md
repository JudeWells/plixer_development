# Plixer — working brief

Full measured history is in **`report.md`** (§1–22) — consult it for detail; this file is the
brief. §22 carries corrections and baselines back from the `generative-poc2mol` branch and
should be read before trusting §12g or any Dice figure below.

Two-stage pocket-conditioned molecule generator: **Poc2Mol** (3D U-Net, protein voxels → ligand
voxels) → **Vox2Smiles** (ViT encoder + GPT-2 decoder, ligand voxels → SMILES).
Context: PhD thesis chapter. Paper: `plixer_ICML_GenBio_2025_version.pdf`.

---

## 0. CURRENT TASK — end-to-end training (branch `end-to-end`)

**Goal: let the language-modelling loss backprop all the way through Poc2Mol**, so the
density is optimised for what the decoder actually needs rather than for reconstruction.
Today the two stages are trained separately and Poc2Mol is frozen.

**Start from these two checkpoints** (both 11-channel v2 — a mismatch here is silent):

| | path | reference number |
|---|---|---|
| Poc2Mol | `checkpoints/poc2mol_v2/poc2mol_v2_11ch_ep576.ckpt` | Dice **0.5027** (§22b) |
| decoder | `checkpoints/s1_v2/s1_v2_11ch_ep14_step247256.ckpt` | stage-3 AUC **0.7522** |

**Beat 0.7522** on `val/likelihood_auc_znorm` (stage 3 with this same decoder). Note the
parameter-free composition readout scores **0.7615** (§14d) — below that, nothing is earning
its keep. Watch that Poc2Mol's own Dice does not collapse on the way.

### ✅ DONE (2026-08-12) — the gradient path is open. Read report.md §23 before touching it.

`src/models/end_to_end.py` (`EndToEndPoc2Smiles`, subclasses `VoxToSmilesModel` so the
likelihood-AUC machinery is inherited verbatim) + `src/data/vox2smiles/end_to_end.py`
(voxelises only) + `configs/experiment/e2e_*.yaml`. Loss is
`LM cross-entropy + voxel_loss_weight × BCEDice`, param groups give the upstream its own LR.
Verified: with `voxel_loss_weight: 0` the LM loss alone puts grad norm **16.6** on Poc2Mol;
severed it is **exactly 0.0**. Analysis: `scripts/adhoc_analysis/e2e_sweep_report.py`.

**Four things that will bite you here, all measured (§23):**

1. 🚨 **0.7522 IS NOT A LIKE-FOR-LIKE TARGET.** Stage 3's val datasets inherit
   `rotate: true, translation: 6.0` (the config overrides only `use_cluster_member_zero`
   and `data_path`), so its validation was STOCHASTIC, against §6. Under deterministic
   validation the *frozen* baseline scores **0.7660**. Re-measure your own baseline; that
   is what arm `e2e_z_frozen` is for.
2. 🚨 **Three incompatible soft-Dice definitions are live.** Same checkpoint, same batch:
   0.467 (per-sample pooled over channels+space — the 0.5027 yardstick), 0.311
   (per-channel pooled), 0.194 (per sample-channel). Always say which.
3. 🚨 **`val_check_interval` counts MICRO-batches.** At `accumulate_grad_batches: 4` a
   `250` means a validation every 62 optimiser steps, and `patience: 12` means 750 steps,
   not 3000. This truncated an entire sweep at ~step 1350 of 4000, before the LR anneal.
   `src/train.py` now warns; multiply by `accumulate_grad_batches`.
4. ⚠️ **Freeze by zeroing the LOSS TERM, never by detaching** — a detached upstream leaves
   parameters out of the backward pass and plain DDP rejects it outright.

Result so far: round 1 (truncated) had **nothing beating the frozen baseline**, every
contrast inside the ±0.02 noise band, and Dice falling in every arm whose upstream moved.
Round 2 therefore tightens the anchor (`voxel_loss_weight: 3.0`) rather than loosening it.

### The original obstacle (now solved — kept for the reasoning): severed three times

Poc2Mol currently runs in `Poc2MolInferenceBuilder`, which is called from
`Vox2SmilesDataModule.on_after_batch_transfer` — a **datamodule** hook, outside the autograd
graph the LightningModule builds. On top of that it runs under `torch.no_grad()` (line ~209),
and `_bind` calls `.eval()` and `requires_grad_(False)` on every parameter. `VoxToSmilesModel`
never receives the Poc2Mol module at all, only finished `pixel_values`.

So end-to-end requires **moving the Poc2Mol forward into `VoxToSmilesModel.training_step`**
(or a wrapper LightningModule owning both). Deleting the `no_grad` alone will not work.

### Footguns

- **`_bind` hard-casts the model to bfloat16** (`resolve_dtype(voxel_config.dtype)`). Fine for
  frozen inference, wrong for training — use autocast and keep master weights fp32.
- **The quality filter blanks labels** for rejected rows (`quality_filter`, `max_poc2mol_loss`).
  Those rows then contribute no LM loss and therefore **no gradient to Poc2Mol**. Start with
  `quality_filter: none`.
- **Half the training mixture is ZINC** (`prob_poc2mol: 0.5`), and those rows carry no pocket,
  so only complex rows train Poc2Mol. Effective batch for the upstream model is half what it
  looks like.
- **Nothing anchors the density.** With only an LM loss, Poc2Mol is free to emit whatever the
  decoder finds convenient and stop being a density model. Keep an auxiliary BCE+Dice term on
  the predicted grid and tune its weight; check `val/poc2mol/*` and Dice throughout.
- **Two learning rates.** Poc2Mol trained at 1e-4, the decoder fine-tunes at 5e-5. Use param
  groups; the pretrained upstream probably wants the smaller one.
- **Memory.** Backprop now spans a 117M-param 3D U-Net at 32³ plus a 172M decoder. Expect to
  cut batch size or add gradient checkpointing; `target_samples_per_batch` keeps the effective
  batch matched (`src/train.py` overwrites `accumulate_grad_batches`).
- **Select on `val/likelihood_auc_znorm`, never `val/loss`** (§18h: they anti-correlate).
- **`data.config.batch_size` does not set the validation batch size** — see §6.

Useful: `scripts/adhoc_analysis/density_diagnostics.py` will tell you whether the density has
drifted (pocket decomposition + stray-density profile) without retraining anything.

---

## 1. Where things stand

**Poc2Mol (regression, BCE+Dice), `checkpoints/poc2mol_v2/poc2mol_v2_11ch_ep576.ckpt`**,
W&B `cath/poc2mol/2ho5efkl`, is the current model. Measured on the FULL HiQBind v2 val split
(1019 pockets) with pooled soft Dice:

| quantity | value |
|---|---|
| `dice(pred, true)` | **0.5027 ± 0.0033** |
| pocket-blind floor (zero pocket) | 0.2073 |
| gain from *any* pocket (shuffled) | +0.0287 |
| gain from the **correct** pocket | **+0.2667** |
| rotation-control win rate, 15°–180° | **0.938–1.000** |
| stray voxels/pocket above 0.5 in empty space | 584 |

Use **0.5027** as the reconstruction yardstick, on the full split. Not §12g's 0.596 — that is
a different checkpoint on 104 PLINDER-panel pockets and is not comparable (§22b). And
**§12g's "prediction overlaps a wrong pose better than the truth" does NOT reproduce** on
this checkpoint: win rate is ≥0.938 at every angle (§22a). Re-check anything justified by it.

**Reproduce any of the above** with
`scripts/adhoc_analysis/density_diagnostics.py --ckpt <ckpt> --out_channels 11`.

**A generative (flow-matching) Poc2Mol was built and refuted** on the `generative-poc2mol`
branch: it lost on reconstruction Dice (0.435–0.47 vs 0.5027), on downstream likelihood AUC
(0.703 vs 0.752) and on Tanimoto (0.147 vs 0.168). Reconstruction metrics are maximised by
the conditional mean, which is what the regression model already emits, so the generative
formulation is penalised structurally. That branch's §22 has the full account and eight
untried ideas; the two worth revisiting are a **hybrid MSE+Dice loss** and
**`voxel_aggregation: sum`**. Do not restart it without reading that first.

**The binding constraint is probably not the decoder.** A parameter-free composition readout
scores 0.7615 on likelihood AUC (§14d) and the trained decoder scores 0.742–0.752 — below it.
Scaling stage-1 4.3× (57k → 247k steps) bought **+0.010 AUC** (§22d).

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
| density diagnostics | `scripts/adhoc_analysis/density_diagnostics.py` | pocket decomposition (blind / shuffled / correct) and stray-density amplitude profile. Model-agnostic; reproduces the 0.5027 baseline |
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
- ⚠️ `pgrep -f "task_name=X"` matches the SHELL RUNNING IT, so `pkill` kills your own command
  part-way through. Collect PIDs into a file first, or put the launch in a script file so the
  pattern is absent from the invoking command line. This has killed two runs.
- A health check must assert a **W&B run URL appeared** in the log. A run started with
  `WANDB_MODE=offline` looks identical in the progress bar and cannot be switched online
  afterwards.
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
- ⚠️ `data.config.batch_size` does NOT control the validation loader — `data.val_batch_size`
  does (optional; `None` keeps the old `min(4, batch_size)`). Any script that counts val
  BATCHES is counting batches of a size it did not choose; this produced a 128-pocket
  baseline compared against 256-pocket numbers (§22b).
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
