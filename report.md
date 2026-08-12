# Plixer — working notes

Two-stage pocket-conditioned molecule generator: **Poc2Mol** (3D U-Net, protein voxels → ligand voxels)
→ **Vox2Smiles** (ViT encoder + GPT-2 decoder, ligand voxels → SMILES). `CombinedProtein2Smiles` chains them.
Paper: `plixer_ICML_GenBio_2025_version.pdf` (GenBio @ ICML 2025).

Context: being written up as a PhD thesis chapter. The focus is re-running training and
strengthening the evaluation, not new modelling.

---

## 1. Environment — two non-obvious gotchas

**`requirements.txt` IS the verified environment** (torch 2.3.1, lightning 2.3.2, transformers 4.42.3,
rdkit 2023.9.6). Don't trust `pip list` inside `../venvPlixer`: that venv was built by `uv` without its
own `pip`, so bare `pip` resolves to a **conda base env** and reports completely different versions.
Always introspect with the venv's own interpreter:

```bash
../venvPlixer/bin/python -c "import importlib.metadata as m; print(sorted((d.metadata['Name'],d.version) for d in m.distributions()))"
```

**Two fixes are required, not optional:**

1. **docktgrid bfloat16 patch.** The README calls it optional. It isn't — every Poc2Mol config hardcodes
   `dtype: torch.bfloat16` while docktgrid ships `DTYPE = torch.float32`, so the random-rotation transform
   dies with `expected m1 and m2 to have the same dtype`. Patch
   `<venv>/lib/python3.11/site-packages/docktgrid/config.py` → `DTYPE = torch.bfloat16`.
   **This is a site-packages edit and does not survive a venv rebuild.**
   ⚠️ Still required for the *legacy* voxelisation path and anything that builds a `MolecularComplex`
   (rotation happens in docktgrid's DTYPE). The new batched voxeliser (§3b) never reads docktgrid's
   globals, so once inference and evaluation are ported off `UnifiedVoxelGrid` this patch can go.
2. **`setuptools==80.9.0`.** Newer setuptools (81+) removed `pkg_resources`, which `lightning_utilities`
   imports at module load. A fresh `uv pip install -r requirements.txt` resolves setuptools 83 and training
   crashes on import. Not currently pinned in `requirements.txt`.

---

## 2. Running training

```bash
export WANDB_MODE=offline          # or `wandb login`
python src/train.py experiment=train_poc2mol_hiqbind
python src/train.py experiment=train_vox2smiles_zinc
python src/train.py experiment=train_vox2smiles_combined_hiqbind
```

- Use `experiment=...`, **not** `+experiment=...` (the latter raises a Hydra composition error).
- `trainer.max_steps` / `limit_*_batches` aren't in the trainer config — add with `+trainer.max_steps=N`.
- `data.num_workers` needs `++` in the combined config (already present) but `+` in the zinc config.
- **`logger=null_logger` is broken** — `LearningRateMonitor` requires a logger. Use `WANDB_MODE=offline`.
- README says `experiment=train_vox2smiles_combined`; the real name is `train_vox2smiles_combined_hiqbind`.
- `configs/trainer/default.yaml` hardcodes `devices: 1`. `ParquetDataset` is map-style so DDP should work,
  but the combined stage's `CombinedDataset` mixes two sources by sampling probability — **untested under DDP**.
- `../geom/rdkit_folder/drugs` is missing but is dead config (only a fallback when `train_dataset`/
  `val_datasets` are unset). Its one live effect: the zinc experiment's *test* split resolves to nothing.

All three stages verified end-to-end on 2026-08-06 (local RTX 3090 and the H100 node).

---

## 3. Data

Configs use paths relative to the repo root, so datasets sit **beside** the repo.

| Path | Contents |
|---|---|
| `../hiqbind/parquet/{train,val,test}` | 793 / 106 / 84 parquet files (955 MB) — Poc2Mol + combined |
| `../zinc20_parquet` | 5,537 files (5.9 GB) — Vox2Smiles pretraining |
| `../hiqbind/raw_data_hiq_sm` | 36 GB raw structures |
| `../PDBbind_v2020_refined-set`, `../validation_PDBbind` | legacy, superseded by HiQBind |

Parquet columns: `system_id, smiles, protein_coords, ligand_coords, split, cluster,
protein_cluster_id, ligand_cluster_id`.

**Gone:** raw ZINC20 mol2 (`../zinc20`) and raw Plinder (`/mnt/disk2/plinder/`). Processed parquet is
intact so retraining is unaffected, but those two can't be regenerated from scratch.

Checkpoints: `python download_checkpoints.py` → HF `judewells/plixer_v1`.

### Nebius node (`ssh dipsae`) — PRIMARY DEV MACHINE
8× H100 80 GB, 128 cores, 1.5 TB RAM. All three training stages verified here 2026-08-06.

```
~/plixer_outer/plixer/          repo + venvPlixer + checkpoints   <- run everything from here
~/plixer_outer/hiqbind/parquet/ 793 / 106 / 84
~/plixer_outer/zinc20_parquet/  5537
```

The `plixer_outer/` wrapper mirrors the local `/mnt/disk2/VoxelDiffOuter/` layout so the configs'
`../hiqbind/...` relative paths resolve. **Development now happens on the node**, not locally.

- Git remote is the **private** repo `git@github.com:JudeWells/plixer_development.git` (not the public
  `JudeWells/plixer`). Node was at 536299e when the private remote was set.
- The venv survived being moved (uv venvs relocate fine; `sys.prefix` resolves from `pyvenv.cfg`).
  Both patches are intact there — docktgrid `bfloat16`, setuptools 80.9.0.
- `/mnt/cloud-metadata` is **read-only** — use home.
- Cross-machine sanity check: Poc2Mol `val/loss` matched the local run to 6 dp (1.660364).

---

## 3b. Voxelisation rewrite + three input-data bugs (2026-08-06)

Prompted by the 8-GPU / protein-channel work. **All three bugs change Poc2Mol's input**, so every
checkpoint trained before this date was trained on different data from what the current code produces.

### The bugs

1. **Protein channel 3 was "not sulfur", not sulfur.** `UnifiedView.get_molecular_complex_channels`
   inverted the last row of *both* channel maps. Correct for the ligand map, whose last entry lists the
   common elements so inverting yields exotics and drops H. Wrong for the protein map, whose last entry is
   just `["S"]` — inverting gave "everything except S". Measured on one complex: channel 3 held
   1650 H + 1044 C + 325 O + 271 N; 1640/3295 protein atoms sat in two channels at once; **protein S was
   in no channel at all**. `remove_hydrogens: true` never helped — it only reaches the RDKit ligand path.
   Independently confirmed in W&B `ljv96zyo`: `channel_mean/protein_3 = 0.283`, the *largest* protein
   channel (carbon is 0.219). If it were really sulfur it would be ~0.001, like `ligand_3`.
   Fixed via `protein_last_channel_is_catch_all: False` (new config flag; ligand flag stays `True`).
   Cost of the fix: 32 Se + 25 P protein atoms per ~960 complexes now fall in no channel. Negligible
   (5e-6 of atoms) but it is the argument for a 5th protein catch-all channel if you ever want one.

2. **Halogen element symbols were case-mismatched between the two data sources.** HiQBind parquet carries
   PDB-style uppercase (`CL`, `BR`, `SE`); RDKit hands back title case (`Cl`, `Br`). The channel maps use
   title case, so over 960 HiQBind complexes all 184 `CL` and 49 `BR` fell through to the ligand catch-all
   while ZINC populated channels 4 and 7 normally. **The chlorine and bromine channels were permanently
   empty for every HiQBind target** — Poc2Mol had no signal to ever emit them, while the ZINC-pretrained
   decoder expected chlorine in channel 4. A train/serve split through the middle of the combined model.
   Fixed by `np.char.title` at the match site; docktgrid already normalises this way for its vdW lookup
   (`ptable[a.title()]`), so title case is the established convention.

3. **Voxelising in absolute PDB coordinates in bfloat16.** The grid was built as `points + center` and
   distances as `atom - grid_point`, all in bfloat16, using raw coordinates. Grid centres sit a median
   28 Å and up to 92 Å from the origin, where bfloat16 spacing is 0.25–1.0 Å — comparable to or coarser
   than the 0.75 Å voxel. Result: **3.67% of occupied voxels wrong by >0.05**, max error 0.61 on a 0–1
   occupancy. Fixed by subtracting the centre in float32 before voxelising, which is exact algebra:
   `atom - (points + center) ≡ (atom - center) - points`. Residual bfloat16-vs-float32 disagreement drops
   from 3.67% to 0.66% of occupied voxels; the new path computes in float32 anyway.

### The new voxeliser

`src/data/common/voxelization/batched.py`. Dataloader workers now do CPU-only prep (parquet → centred
coords, vdW radii, per-channel atom masks) and the grid is built batched **on the rank's own device**.

This was the actual DDP blocker: `UnifiedVoxelGrid` allocates on `docktgrid.config.DEVICE`, which is a
bare `torch.device("cuda")` — the *current* device. Every worker voxelised on `cuda:0`, which is why
`num_workers: 0` is hardcoded everywhere, and why 8 ranks would have piled onto one GPU.

Three representation choices, in order of how much they bought:

| | ms/sample | notes |
|---|---|---|
| legacy, per-sample on `cuda:0` | 3.6 | but forces `num_workers=0` → 25 samples/s end-to-end |
| naive batched (padded, dense) | 22.5 | padding to batch-max atoms costs more than batching saves |
| + ragged layout, drop unchannelled atoms | 2.47 | ~half of every complex is H, now in no channel |
| + local-neighbourhood kernel | — | occupancy decays as d⁻¹²; each atom touches a small cube |
| **+ box-reach filter** | **0.66** | callers prune to 32 Å but the box is ±12 Å |

The box-reach filter is exact, not an approximation: an atom further than `half_extent + cutoff·vdw` from
the centre along any axis cannot reach any voxel's neighbourhood. Atoms/sample: ~3300 → ~900.

`tests/test_batched_voxelizer.py` proves each claim separately — float32 arithmetic vs an independent
float64 CPU implementation (6e-7), neighbourhood truncation vs its analytic bound (1.09e-4 vs 2.4e-4,
36× under the bfloat16 storage resolution), equivalence with the legacy voxeliser on identical input,
batching invariance. Run it after touching anything in `voxelization/`.

**Benchmarks that set the design** (H100, measured 2026-08-06): Poc2Mol trains at 869 samples/s,
Vox2Smiles at 763; the old data pipeline delivered 25. CPU voxelisation is not an option — 2249 ms/sample
against 3.6 ms on GPU. Going 9→14 ViT input channels costs nothing (172.40M → 172.65M params, identical
throughput), so protein-channel injection is free at the model level.

---

## 3c. Poc2Mol's `val/loss` has a large irreducible floor (measured 2026-08-06)

**A perfect prediction scores 0.6256 on this loss.** Runs converge to ~0.827, so roughly **76% of the
reported number is a constant no model can improve**. It is 98.7% Dice (BCE contributes ~0.011).

Cause: `compute_per_channel_dice` averages over all 9 ligand channels, but only **3.59 of 9 are
occupied** in a typical ligand. For an all-zero target channel
`dice_coef = 2·0 / (Σpred² + 0) = 0` regardless of the prediction, contributing a full 1.0.

| ch | element | occupied in | loss when prediction is **exact** |
|---|---|---|---|
| 0 | C | 100% | 0.027 |
| 1 | O | 100% | 0.036 |
| 2 | N | 90.6% | 0.126 |
| 3 | S | 43.8% | 0.579 |
| 4 | Cl | 9.4% | 0.910 |
| 5 | F | 0% | 1.000 |
| 6 | I | 0% | 1.000 |
| 7 | Br | 3.1% | 0.970 |
| 8 | other | 12.5% | 0.879 |

**Training is not harmed.** An empty target channel gives `dice_coef ≡ 0` with *zero gradient*, so it
is a pure additive constant, not a misleading learning signal. Two things it does break:

1. **Model selection reads as less discriminative than it is.** A sweep spanning 0.823–0.834 looks like
   a 1.3% spread; against actual model error (loss − floor) it is 0.198–0.209, a **5.5% spread**.
   Ranking is unaffected (constant offset) but effect sizes are ~3.5× larger than they appear.
   Subtract the floor, or compute Dice over occupied channels only, before comparing configs.
2. **`max_poc2mol_loss` is measuring ligand composition, not reconstruction quality.** ⚠️ The
   per-sample floor ranges **0.473 to 0.802** purely with how many channels that ligand occupies. An
   absolute threshold of 0.79 therefore rejects a halogen-free ligand *however perfectly reconstructed*,
   and accepts a chemically rich one that was reconstructed badly. As a quality gate it is close to
   noise, and it silently biases which pockets the decoder trains on.

### What to do about `max_poc2mol_loss` — now a config choice

`data.quality_filter` in the combined configs, implemented in `Poc2MolInferenceBuilder`:

| value | behaviour |
|---|---|
| `none` | **default.** Train on every Poc2Mol output; `max_poc2mol_loss` ignored. |
| `relative` | Reject when the loss exceeds *that sample's own floor* by `max_poc2mol_loss`. Also logs `poc2mol_excess_loss`. |
| `absolute` | Legacy. Kept only to reproduce old runs — do not use for new work. |

`poc2mol_loss_floor()` computes the floor by scoring a perfect prediction of the target;
it is a handful of elementwise ops, no model forward.

**Removing it entirely is the leading option** and is more defensible than it first looks: at inference
the decoder must handle whatever Poc2Mol produces, so filtering training down to the easy
reconstructions creates a train/serve mismatch — the decoder never learns the hard pockets. The
counter-argument is that training on hopeless reconstructions teaches hallucination (a confident SMILES
unrelated to the input density). If a gate is wanted, it must be relative: threshold the *excess* over
each sample's own floor, or compute Dice over occupied channels only. **Do not** carry the inherited
absolute 0.79 forward — under the corrected channel encoding it rejects 100% of samples anyway.

---

## 3d. Poc2Mol sweep 1 — hyperparameters barely matter (2026-08-06, killed at ~14k steps)

W&B group `exp1_poc2mol_20260806_135055`, project `cath/poc2mol`. **Killed deliberately at
~14k steps** once it was clear the axes were not discriminating; superseded by §3e.

| run | lr | wd | dropout | best `val/loss` | @epoch | W&B |
|---|---|---|---|---|---|---|
| `base` | 1e-4 | 0.2 | 0.1† | 0.8155 | 136 | [oapzu6h6](https://wandb.ai/cath/poc2mol/runs/oapzu6h6) |
| `lr_low` | 3e-5 | 0.2 | 0.1† | 0.8173 | 145 | [uvi28tqc](https://wandb.ai/cath/poc2mol/runs/uvi28tqc) |
| `lr_high` | 3e-4 | 0.2 | 0.1† | 0.8158 | 109 | [hihzt2no](https://wandb.ai/cath/poc2mol/runs/hihzt2no) |
| **`wd_low`** | 1e-4 | **0.05** | 0.1† | **0.8137** | 161 | [urb2qebe](https://wandb.ai/cath/poc2mol/runs/urb2qebe) |
| `wd_high` | 1e-4 | 0.5 | 0.1† | 0.8138 | 161 | [aqzcdjkq](https://wandb.ai/cath/poc2mol/runs/aqzcdjkq) |
| `drop_none` | 1e-4 | 0.2 | 0.0† | 0.8142 | 145 | [hox8q6r4](https://wandb.ai/cath/poc2mol/runs/hox8q6r4) |
| `drop_high` | 1e-4 | 0.2 | 0.2† | 0.8142 | 145 | [you7im7m](https://wandb.ai/cath/poc2mol/runs/you7im7m) |
| `reg_combo` | 3e-4 | 0.5 | 0.2† | 0.8147 | 183 | [nkwtxyeq](https://wandb.ai/cath/poc2mol/runs/nkwtxyeq) |
| `wd_zero_constlr` | 1e-4 | **0.0**, constant LR | 0.1† | (still running) | | [5myk5h76](https://wandb.ai/cath/poc2mol/runs/5myk5h76) |

† **dropout was a no-op in all of these.** `nn.Dropout` is only inserted when `layer_order`
contains `'d'`, and every config used `'gcr'`. So `base`, `drop_none` and `drop_high` were the
*same* configuration — and produced bit-identical curves, which incidentally proves the
pipeline is exactly reproducible across processes and GPUs. Three of eight slots tested nothing.

**Findings:**

1. **lr, weight decay and dropout are all within 0.003 of each other.** Only lr 3e-5 is clearly
   bad (+0.0023). Nothing else separates. The optimiser is not the binding constraint.
2. **The model overfits: train real error ~0.065 vs val ~0.179, a 2.75× gap that widens.** More
   capacity will not help; regularisation and representation are the levers. Note this makes the
   never-actually-enabled dropout an untested and promising axis.
3. ⚠️ **The historical run `ljv96zyo` ends *better* than any of these.** Its minimum is
   **0.8034 at epoch 173** (real error 0.1663) vs our best 0.8137 (0.1787). It was *behind*
   early — 0.8280 at epoch 60 vs our 0.8156 — but kept improving while ours plateaued.
   Three confounded candidate causes: (a) constant LR vs our cosine decay, (b) wd 0 vs our
   ≥0.05, (c) **the protein encoding fix removed hydrogens.** `wd_zero_constlr` tests (a)+(b);
   §3e tests (c).
   **→ (c) is now the leading explanation, and it is leakage, not representation quality:**
   HiQBind's protein hydrogens are positioned with the ligand present, and the old channel 3 was
   predominantly hydrogen. See **§3f**.
4. `ljv96zyo` itself overfits mildly: +0.0077 from its epoch-173 minimum over the next 220
   epochs. So ~epoch 175 is the expected location of the turn.
5. **Capacity options on a 32³ grid**: `f_maps` 32/64/96/128 → 29/117/264/469M params;
   `num_levels` 4/5/6 → 29/117/470M. **`num_levels ≥ 7` crashes** — the grid is exhausted after
   5 downsamples. `scripts/tune_poc2mol_optuna.py` suggests `num_levels ∈ {5,7,9,11}` and would
   therefore fail on three of its four options.

---

## 3e. Poc2Mol sweep 2 — protein representation (launched 2026-08-06)

W&B group `exp1_poc2mol_protein_repr_<stamp>`. Launcher
`scripts/run_exp1_poc2mol_protein_repr.sh`; configs `configs/experiment/exp1_poc2mol_{cons4,h5,all5,hall6}.yaml`.

Four protein encodings × {dropout off, dropout **actually on** via `layer_order: gcrd`} = 8 runs,
one per GPU, hyperparameters fixed at sweep 1's best (lr 1e-4, wd 0.05).

| variant | protein channels | `in_channels` | dropout off | dropout 0.1 |
|---|---|---|---|---|
| `cons4` | C, O, N, S | 4 | [u0t1gnlw](https://wandb.ai/cath/poc2mol/runs/u0t1gnlw) | [m29u9gom](https://wandb.ai/cath/poc2mol/runs/m29u9gom) |
| `h5` | C, O, N, S, **H** | 5 | [70jru5gv](https://wandb.ai/cath/poc2mol/runs/70jru5gv) | [bfgkknmr](https://wandb.ai/cath/poc2mol/runs/bfgkknmr) |
| `all5` | C, O, N, S, **ALL** | 5 | [k04ibc62](https://wandb.ai/cath/poc2mol/runs/k04ibc62) | [c6shh1mm](https://wandb.ai/cath/poc2mol/runs/c6shh1mm) |
| `hall6` | C, O, N, S, **H**, **ALL** | 6 | [jc8z8mbc](https://wandb.ai/cath/poc2mol/runs/jc8z8mbc) | [tqfw0sb8](https://wandb.ai/cath/poc2mol/runs/tqfw0sb8) |

`ALL` uses a new `"*"` wildcard in the channel map (`UnifiedView`), matching every atom, so a
generic total-density channel is expressible without enumerating the periodic table.

### Two changes that break comparability with earlier runs

1. **Validation is now deterministic.** It previously inherited `random_rotation: true` and
   `random_translation: 6.0`, *and* `ParquetDataset` draws a random cluster member per epoch — so
   every validation scored a different augmentation of a different molecule. `val/loss` was a
   moving target, which is why sweep 1's curves were noisy and "has it overfitted" was hard to
   call. Now `rotate: false, translation: 0.0, use_cluster_member_zero: true` for val **and**
   test. ⚠️ **Absolute `val/loss` values are not comparable with anything before 2026-08-06**,
   `ljv96zyo` included; re-evaluate old checkpoints on this setting to bridge.

   **Bridge measured 2026-08-06**, one checkpoint (`wd_low` @ep161) under all four
   combinations, full 1019-system val:

   | setting | val/loss | floor | real error |
   |---|---|---|---|
   | (a) random member + augmentation — the OLD setting | 0.8389 | 0.6400 | 0.1989 |
   | (b) random member, no augmentation | 0.8313 | 0.6398 | 0.1915 |
   | (c) member-zero + augmentation | 0.8384 | 0.6404 | 0.1981 |
   | (d) member-zero, no augmentation — the NEW setting | 0.8313 | 0.6403 | 0.1909 |

   **Deterministic val is EASIER by −0.0076**, entirely from dropping augmentation;
   `use_cluster_member_zero` is worth −0.0005 and the floor is essentially unchanged
   (+0.0004). The irreducible floor on the deterministic val is **0.6403**.

   ⚠️ **The far bigger effect is selection bias, not the val setting.** Sweep 1's reported
   bests are *minima of a noisy series*: the `wd_low` checkpoint W&B logs as **0.8137**
   actually scores **0.8389** in expectation on that same stochastic setting — an optimistic
   bias of ~**+0.025**, three times larger than the setting change and in the opposite
   direction. Any "best val/loss" from a stochastic-val run (sweep 1 **and `ljv96zyo`**) is
   inflated by roughly this much and must not be quoted against sweep 2's deterministic
   numbers. This is the real argument for fixing the val set.

   Consequence for §3d finding 3: `ljv96zyo`'s 0.8034 carries the same bias, so
   "the historical run beats ours" was measured like-for-like — but both figures are
   optimistic, and the comparison should be redone from checkpoints on the deterministic val.

   Re-scored honestly on the deterministic val: `wd_high` @ep161 **0.8291** (real 0.1887),
   `wd_low` @ep161 0.8313 (0.1909), `base` @ep112 0.8324 (0.1920) — so **wd 0.5 edges wd
   0.05**, where the noisy metric had them tied the other way. Sweep 2 runs wd 0.05; left as
   is because weight decay is constant across all 8 arms and so cannot confound the
   representation comparison, but wd 0.5 is the better choice for the final run.
2. **`batch_size` 64 → 128, accumulation 2 → 1** (same effective batch 128). Sweep 1 used only
   14 GB of an 80 GB card; this uses ~22 GB.

⚠️ **The `h5` and `hall6` arms are confounded by ligand-conditioned protonation — see §3f.**
Their explicit hydrogen channel is partly a channel to the target, so a `val/loss` win for those
arms cannot be read as a representation improvement without the diagnostics in §3f.

---

## 3f. HiQBind protein hydrogens are ligand-conditioned — leakage into any H channel (2026-08-06)

Prompted by the observation that Poc2Mol scores better with explicit protein hydrogen channels.
**It does, but the hydrogens were positioned using the ligand**, so a protein H channel leaks the
answer into a model whose task is protein voxels → ligand voxels.

Chain of custody through HiQBind-WF (`github.com/THGLab/HiQBind`):

1. `process.py:754` — the refinement entry point is literally `refine_structure_with_ligand`, and
   builds `StandardizedPDBFixer(protein_pdb=..., ligand_sdf=...)`.
2. `fix_protein.py:343-361` — `addLigand()` merges the ligand into the fixer's topology
   *permanently*, before anything else runs (`self.topology = modeller.getTopology()`, line 360).
3. `fix_protein.py:632-633` — `self.addMissingHydrogens(7.4)` therefore runs on a topology that
   contains the ligand.
4. `scripts/create_hiqbind_dataset.py:123` — the parquet is built from `*_protein_refined.pdb`,
   the output of exactly this. Measured: HiQBind protein atoms are **49.8% H**.

### Two distinct mechanisms

**1. HIS tautomer selection.** PDBFixer delegates to OpenMM `Modeller.addHydrogens`, whose rule 3
is *"for a neutral Histidine residue, the HID or HIE variant is selected based on which one forms a
better hydrogen bond."* Its acceptor list (`modeller.py:881`) is built from **the entire topology**,
and the only exclusion (line 943) is the histidine itself — so ligand O/N atoms directly determine
pocket HIS tautomers.

**2. Every hydrogen is energy-minimised in the ligand's force field — the bigger effect.**
`refineAddedAtomPositions` (`fix_protein.py:550-594`) pins atoms by zeroing their mass, but line 581
reads `if atom.element is app.element.hydrogen: continue` — hydrogens are *never* mass-zeroed. So
**all heavy atoms are frozen and every hydrogen is free**, and `LocalEnergyMinimizer.minimize`
(line 594) then runs on a system whose force field explicitly includes the ligand (SMIRNOFF, GAFF
fallback, Gasteiger charges — lines 666-678). Every rotatable polar hydrogen in the pocket
(Ser/Thr/Tyr OH, Lys NH₃⁺, Asn/Gln NH₂, backbone NH) is relaxed toward ligand acceptors.

### How much of the h5 gain is leakage — unknown, needs measuring

Do not over-claim. Nonpolar C–H hydrogens are geometrically pinned by their frozen parent heavy
atoms and barely move, so they carry legitimate shape/occupancy signal. The ligand-conditioned
subset is the polar/rotatable hydrogens — a minority of all H, but concentrated exactly at the
pocket, which is where the Dice loss is computed. The h5 advantage is therefore a **mixture** of
real signal and leakage in unknown proportion.

### This resolves the §3d finding-3 puzzle

§3d flagged that historical run `ljv96zyo` ends better (0.8034) than any sweep-1 run (0.8137), with
candidate cause "(c) the protein encoding fix removed hydrogens". §3b bug 1 established that the old
protein channel 3 was "not sulfur" and held ~1650 H per complex — i.e. **the released model's
channel 3 was predominantly hydrogen.** If those hydrogens are ligand-conditioned, the released
model was consuming leakage, and correcting the encoding would *look* like a regression while
actually being a fix. That matches the observed pattern, and is the leading explanation.

### Consequence regardless of how the split comes out

Whatever fraction is genuine signal, **it is not available prospectively** — on a novel target you
protonate without knowing the ligand. If `h5`/`hall6` goes forward, the pipeline must protonate
ligand-blind at both train and inference, or it has a train/serve mismatch of exactly the kind
§3c warns about for `max_poc2mol_loss`.

Related: the Runs N' Poses eval set (§6, `BENCHMARK_PLAN.md` §5a-i-b) uses **unprotonated** deposited
receptors, so it is a fair prospective test of this — sharp `h5` degradation there relative to
`cons4` would corroborate. It also means `h5`/`hall6` cannot be scored on that set as-is.

### Diagnostic 1 RESULT — no detectable pocket-specific H orientation bias (2026-08-06)

`scripts/adhoc_analysis/diagnose_h_leakage.py`, 300 val complexes. Every protein H is assigned to
its nearest heavy atom (bond lengths separate cleanly: H-C 1.09, H-N 1.01, H-O 0.97 A), giving
polar (N/O/S-bound, rotatable) vs nonpolar (C-bound, pinned by its frozen parent). Metric is
`d(parent -> nearest ligand acceptor) - d(H -> same)`: positive means the H points at the ligand.
**Nonpolar C-H is the built-in null — it cannot rotate.**

| group | n | mean(dParent − dH) | % pointing at ligand |
|---|---|---|---|
| polar, pocket | 6,932 | **+0.0391 Å** | 50.7% |
| nonpolar, pocket | 27,533 | −0.0012 Å | 49.3% |
| polar, distal | 202,921 | −0.0128 Å | 48.9% |
| nonpolar, distal | 710,840 | −0.0560 Å | 46.2% |

Polar pocket H do point ligand-ward more than polar distal H — **but nonpolar H shift by the same
amount**, so that is geometry (all H point outward from the core, which near the pocket means
toward the ligand), not chemistry. The polar-specific excess is **+0.0403 Å in the pocket vs
+0.0432 Å distally** — indistinguishable, marginally larger *away* from the ligand.
H-bond contacts: 9.02% of pocket polar H within 2.6 Å of a ligand acceptor vs 1.52% nonpolar,
which is ~2 H-bonds per complex — ordinary chemistry.

⚠️ **This is not exoneration.** Two limits:
- **Blind to mechanism 1.** HIS tautomer choice changes *which* nitrogen carries the H — a
  presence/absence change. A displacement metric cannot see it.
- **Limited power.** If only the ~9% of pocket polar H that H-bond the ligand were reoriented, the
  expected mean shift is ~0.045 Å — the same order as measured. "No conditioning" and "conditioning
  confined to the H-bonding minority" are not separated.

What it does argue against is *broad* reorientation of pocket polar hydrogens, i.e. mechanism 2
acting at scale. Consistent with a local minimiser barely moving H from good starting geometry.

### Diagnostic 3 RUNNING — polar/nonpolar H ablation

W&B group `exp1_poc2mol_hleak_20260806_194134`. New channel tokens `H_polar` / `H_nonpolar` in
`UnifiedView` (nearest-heavy-neighbour parent assignment; 69 polar / 204 nonpolar per complex,
summing to `h5`'s 273).

| arm | channel 5 | reads as | W&B (nodrop / drop) |
|---|---|---|---|
| `hpolar5` | H bonded to N/O/S — rotatable | recovers the h5 gain → **leakage** | [svrzf91m](https://wandb.ai/cath/poc2mol/runs/svrzf91m) / [l0d1o1m4](https://wandb.ai/cath/poc2mol/runs/l0d1o1m4) |
| `hnonpolar5` | H bonded to C — pinned | recovers the gain → legitimate surface detail | [osn7eajs](https://wandb.ai/cath/poc2mol/runs/osn7eajs) / [8rr2svam](https://wandb.ai/cath/poc2mol/runs/8rr2svam) |
| `heavyall5` | aggregate heavy density, no H | controls for "a 5th aggregate channel helps regardless" | [45plwnmb](https://wandb.ai/cath/poc2mol/runs/45plwnmb) / [a0lviodl](https://wandb.ai/cath/poc2mol/runs/a0lviodl) |

Caveat: polar (69 atoms) and nonpolar (204) channels differ in density as well as chemistry, so a
**null** result on `hpolar5` is weaker evidence than a positive one. `heavyall5` partly absorbs this.

### Sweep-2 arms killed for H contamination (2026-08-06)

| run | best `val/loss` | real error | W&B |
|---|---|---|---|
| `h5_drop` | 0.7956 | **0.1553** | [bfgkknmr](https://wandb.ai/cath/poc2mol/runs/bfgkknmr) |
| `h5_nodrop` | 0.7966 | 0.1563 | [70jru5gv](https://wandb.ai/cath/poc2mol/runs/70jru5gv) |
| `hall6_nodrop` | 0.7966 | 0.1563 | [jc8z8mbc](https://wandb.ai/cath/poc2mol/runs/jc8z8mbc) |
| `hall6_drop` | 0.7972 | 0.1569 | [tqfw0sb8](https://wandb.ai/cath/poc2mol/runs/tqfw0sb8) |
| `all5_nodrop` | 0.7983 | 0.1580 | [k04ibc62](https://wandb.ai/cath/poc2mol/runs/k04ibc62) |
| `all5_drop` | 0.8006 | 0.1603 | [c6shh1mm](https://wandb.ai/cath/poc2mol/runs/c6shh1mm) |

`all5` was killed too: its `ALL` channel is `["*"]`, so hydrogens are present, merged rather than
explicit — same leakage path. Surviving clean arms: `cons4_drop` **0.8039** (real 0.1636, best clean
result), `cons4_nodrop` 0.8057, plus `wd_zero_constlr`. **So ~5% of Poc2Mol's real error is the
amount at stake** in this question.

### Remaining diagnostics (blocked on this node)

Cheap → decisive:

1. **H-bond geometry test** (hours, no retrain) — fraction of pocket hydroxyl/amine hydrogens
   sitting in H-bond geometry with a ligand acceptor, versus the same proteins protonated
   ligand-blind. Gives a direct magnitude.
2. **HIS tautomer skew** (hours, no retrain) — HID/HIE ratio for pocket vs distal histidines.
   Ligand-blind protonation should show no pocket/distal difference.
3. **Polar-vs-nonpolar H ablation** (needs a retrain) — put only carbon-bound hydrogens in the H
   channel. Gain survives → shape; gain collapses → leakage.
4. **Re-protonate apo and retrain** (decisive) — `_has_ligand=False` is already a supported code
   path, so it is a one-argument change (`ligand_sdf=None`) on the same `*_protein.pdb` inputs.
   ⚠️ **`../hiqbind/raw_data_hiq_sm` is not on the node** — only `../hiqbind/parquet` — so this
   needs the 36 GB raw structures from the local machine plus the HiQBind conda env.

---

## 3g. Poc2Mol DECIDED — `cons4_drop`, no hydrogen channels (2026-08-06)

**Shared frozen Poc2Mol for both decoder arms:**
`checkpoints/exp1_shared_poc2mol/poc2mol_cons4_drop_epoch307.ckpt`
(epoch 307, global_step 23716, `val/loss` **0.8039**, real error **0.1636**, provenance commit
536299e19703). Config: protein C/O/N/S, `layer_order: gcrd` + dropout 0.1, lr 1e-4, wd 0.05,
effective batch 128, deterministic val.

**Chosen without hydrogen channels despite hydrogens scoring better**, because the gain is not
usable prospectively — see §3f. Diagnostic 3 settled it:

| arm | channel 5 | atoms/complex | best val | real err | % of h5 gain |
|---|---|---|---|---|---|
| `cons4` | — | 0 | 0.8057 | 0.1654 | 0% |
| `heavyall5` | aggregate heavy, no H | 315 | 0.8055 | 0.1652 | **2%** |
| `hnonpolar5` | C-bound H (pinned) | 204 | 0.8041 | 0.1638 | **17%** |
| **`hpolar5`** | N/O/S-bound H (rotatable) | **69** | 0.7975 | 0.1572 | **82–94%** |
| `h5` | all H | 273 | 0.7970 | 0.1567 | 100% |

**69 rotatable polar hydrogens carry 82–94% of the entire hydrogen benefit; 204 pinned nonpolar
hydrogens carry 8–17%; a hydrogen-free aggregate channel carries ~0%.** The density confound runs
the *wrong way* for an innocent explanation — the sparser channel carries ~5× the signal — so this
is not about occupancy or surface detail. And since `cons4` already contains every N and O position,
the only new information in a polar-H channel is **bond orientation**, which is exactly what
HiQBind's minimiser sets with the ligand in the force field and all heavy atoms frozen.

Residual caveat, unresolved: backbone amide N–H is polar but *fixed* by peptide geometry, so part of
the `hpolar5` signal is legitimate. The sharper test would split O-bound H (hydroxyls, essentially
all freely rotatable) from N-bound H — not run.

Also resolved: `wd_zero_constlr` (constant LR, wd 0) reached real error 0.1755, **worse** than
`cons4_drop` — so §3d candidate causes (a) and (b) are rejected and (c), the hydrogen removal, is
confirmed as the explanation for `ljv96zyo` appearing to beat the corrected runs.

### Overfitting answer (the first run to complete 600 epochs)

`cons4_drop`: minimum **0.8044 at epoch 311**, final 0.8161 at epoch 599 — **drift +0.0118 over 288
epochs**. Train falls monotonically while val turns:

| epochs | min val | mean train | gap |
|---|---|---|---|
| 100–200 | 0.8071 | 0.6499 | +0.162 |
| **300–400** | **0.8044** | 0.5777 | +0.231 |
| 500–600 | 0.8080 | 0.5289 | **+0.284** |

Last 100 epochs: val +0.0012/100 while the gap widens +0.0200/100. **~300–350 epochs is the useful
budget for Poc2Mol, not 600.**

---

## 3h. Experiment 1 decoder — 3-stage curriculum (stage 1 launched 2026-08-06)

Design (Jude's): decouple "learn to use the protein channels" from "learn to cope with a noisy
upstream", rather than confounding them in one fine-tuning stage.

| stage | data | ligand voxels | protein channels |
|---|---|---|---|
| 1 | ZINC only | ground truth | present but empty, flag 0 |
| 2 | + HiQBind complexes | **ground truth** | real, masked 25% of the time |
| 3 | same mix | **ramp true → Poc2Mol prediction** | real, masked 25% |

Stage 3's ramp is `Poc2MolInferenceBuilder.predicted_fraction(global_step)`, linear over
`[predicted_ramp_start_step, predicted_ramp_end_step]`. Validation always uses the prediction
(the deployed condition) so the metric does not drift as the ramp advances.

**Stage 1 running:** W&B group `exp1_stage1_zinc_20260806_224140` — baseline 9ch
[vcbhj5kn](https://wandb.ai/cath/voxelSmiles/runs/vcbhj5kn), protein 14ch
[sznmqndb](https://wandb.ai/cath/voxelSmiles/runs/sznmqndb). 4 GPUs each under DDP, effective batch
256, identical data/schedule/seed — only the channel count differs.

⚠️ **ZINC train/val previously pointed at the same `index.csv`**, so validation was drawn from the
training molecules and `val/loss` measured memorisation. Now split at file level: 60 of 5537 files
held out (seeded permutation) → `index_train.csv` (8,859,304 molecules) / `index_val.csv` (104,029),
provably disjoint. Any ZINC `val/loss` from before 2026-08-06 is not a generalisation measure.

### Configs and throughput

Stages 2 and 3 are written and smoke-tested for both arms:
`configs/experiment/exp1_s{2,3}_{baseline,protein}.yaml`. The frozen upstream is
`configs/model/poc2mol_frozen.yaml` (4 channels, `gcrd`+dropout 0.1, matching the §3g checkpoint's
architecture so `describe_checkpoint` stays honest).

Measured stage-1 rate: **~790 samples/s per arm** (3.09 vs 2.98 it/s, well matched), one ZINC epoch
= 34,607 steps ≈ 3 h. Data-bound, as expected — ZINC's cost is RDKit mol-block parsing, and 2 arms ×
4 ranks × 14 workers already saturates the 128 cores.

### Evaluation harness

`evaluations/evaluate_exp1_arm.py` — one arm → JSON. Reports teacher-forced loss, token accuracy,
validity, uniqueness, Tanimoto to the true ligand, and likelihood AUC both raw and **z-normalised**
(§5.1: raw is 84% ligand-size nuisance and a pocket-blind baseline scores exactly 0.500, so quote
the z-normalised figure). `--mask_protein` runs the protein arm with its pocket channels zeroed —
if that recovers baseline performance the model is genuinely using the protein; if it does not
degrade, the channels are being ignored and any win came from elsewhere.

Positives in the AUC are matched **by SMILES identity, not row index**, because 27 test systems
share a SMILES with another (§5.1) and index-based labelling would score those duplicates as misses.

---

## 3i. Vox2Smiles generation used the wrong end token (fixed 2026-08-07)

`VoxToSmilesModel.__init__` wired the decoder's special tokens to the wrong ids — present since
`ebcdcd0`, the commit that first imported the voxmiles source, so **every run ever done**:

| | was | should be (and now is) |
|---|---|---|
| GPT2 `bos_token_id` | `cls_token_id` = 1 `[CLS]` | `bos_token_id` = 2 `[BOS]` |
| GPT2 `eos_token_id` | `sep_token_id` = 4 `[SEP]` | `eos_token_id` = 3 `[EOS]` |
| VisionEncoderDecoder `pad_token_id` | `eos_token_id` = 3 `[EOS]` | `pad_token_id` = 0 `[PAD]` |

Training is **unaffected** — the loss path masks labels itself using `tokenizer.pad_token_id` and
never reads these. `shift_tokens_right` does fill -100 positions with the config's pad, but
`decoder_attention_mask` already zeroes them, so correcting it does not perturb training either.
The config is rebuilt from the tokenizer on load, so **the fix repairs existing checkpoints
retroactively and no retraining is needed.**

Generation *was* affected: `generate` waited for a `[SEP]` the model never emits, so it ran to
`max_length=200` every time and appended untrained garbage after a complete SMILES.

### But the published results are NOT affected — verified

| model | config | gen length | validity (64 held-out ZINC, greedy) |
|---|---|---|---|
| released `epoch_000.ckpt` | original | 200 | **64/64 = 100%** |
| released `epoch_000.ckpt` | fixed | 68 | **64/64 = 100%** |
| our stage-1 @89k steps | original | 200 | ~0–2% |
| our stage-1 @89k steps | fixed | 65 | **63/64 = 98.4%** |

A converged model emits `[EOS]` and then *keeps* emitting it, and those are stripped by
`skip_special_tokens=True` — so the bug is **self-cancelling once trained**. It only bites
under-trained models whose post-`[EOS]` behaviour has not settled. Published validity, uniqueness,
diversity, QED and LogP therefore stand.

Worth having anyway: generation is ~3x shorter (68 vs 200 tokens), so anything doing bulk
generation — the eval harness especially — gets ~3x faster, and `val/validity` becomes a usable
signal early in training rather than noise.

### How `val/validity` is actually measured

Greedy decode (NOT sampling) of the val-set voxelised ligand, then `Chem.MolFromSmiles`.
`validation_step` calls `generate_smiles(..., max_attempts=1)`, and inside,
`current_do_sample = do_sample or max(attempts) > 0` is False on the first attempt — sampling only
ever occurs on *retries*, which validation never does. The sample is `pixel_values[:30]` from
**batch 0 only**, per rank (~120 molecules), so the epoch-to-epoch swing is sampling noise.

---

## 4. Reproducibility of the paper's tables — IMPORTANT

Everything reproduces exactly, **but the two tables come from two different checkpoints.**

| Paper | Reproduced | Source run |
|---|---|---|
| Table 2 likelihood AUC: chrono 0.67 / PLINDER 0.58 / seq-sim 0.61 | 0.6682 / 0.5834 / 0.6117 ✓ | `evaluation_results/checkpoints_model_run_2025-07-02*` |
| Table 1 sim-enrich 7.58 / 5.50, diversity 0.86 / 0.88, QED 0.60 / 0.56, LogP 0.70 / 0.45 | all ✓ to the digit | `evaluation_results/bubba_zjhnye4j_2025-05-11_highPropPoc2Mol` |
| Vina score (Table 1) and AutoDock Vina AUC 0.62 (Table 2) | ✗ **irreproducible** | `evaluation_results/autodock_vina/` deleted; no `docking_summary.csv` survives |

The two runs are demonstrably different models — on the same chrono split, May gives likelihood AUC
0.6135 and EF 7.58 / diversity 0.86, while July gives 0.6682 and EF 6.32 / diversity 0.99.

The run name suggests July loaded from `checkpoints/` (the released HF weights) while `bubba_zjhnye4j`
is a W&B run id for an unreleased training run — **this is inferred, not verified**; neither directory
stores a config or log recording the checkpoint path. Testable: rerun the eval against `checkpoints/`
and see whether it lands on 0.5834 / 0.6682 / 0.6117.

**Before quoting any number in the thesis, regenerate both tables from one checkpoint.**

Reported AUC convention = **mean of per-system AUCs**, consistently for all three splits
(`calculate_autodock_vina_roc_auc.py` does exactly this; July-run means are 0.6682 / 0.5834 / 0.6117 vs
published 0.67 / 0.58 / 0.61). Per-system *medians* are much higher (0.739 / 0.623 / 0.686) — don't
confuse the two.

---

## 4b. W&B runs behind the released checkpoints

Entity `cath`; projects `poc2mol` and `voxelSmiles`. Credentials are in `~/.netrc`, so
`wandb.Api()` works locally. Local `wandb/` only holds Jan–Feb runs; everything else is server-side.

**Poc2Mol** — released `checkpoints/poc_vox_to_mol_vox/epoch_173.ckpt` (epoch 173, global_step 13572):

| | |
|---|---|
| Run | **`ljv96zyo`** — https://wandb.ai/cath/poc2mol/runs/ljv96zyo |
| Name / host | `poc2mol_HiQBind_kasp` on **kaspian**, started 2025-04-21T17:24:55Z, finished, 28.6 h |
| Reached | epoch 394 — the released ckpt is an **intermediate** checkpoint from this run |
| Args | `experiment=train_poc2mol_hiqbind data.num_workers=3 data.config.batch_size=2 +trainer.num_sanity_val_steps=0 trainer.val_check_interval=null model.lr=…` |
| Evidence | **Definitive** — its console log records `logs/poc2mol/runs/2025-04-21_18-13-26`, the exact path the combined model's config references |

**Combined** — released `checkpoints/combined_protein_to_smiles/epoch_000.ckpt` (epoch 0, global_step
**3,960,000**). Three-run resume chain, each link confirmed by the successor's recorded `ckpt_path`:

| # | Run | URL | Host / start | End global_step |
|---|---|---|---|---|
| 1 | **`crz11hbc`** | https://wandb.ai/cath/voxelSmiles/runs/crz11hbc | kaspian, 2025-03-22T21:19:22Z, finished, 116 h | 2,656,049 |
| 2 | **`55vlvc7x`** | https://wandb.ai/cath/voxelSmiles/runs/55vlvc7x | bubba-213-2, 2025-05-06T19:51:55Z, crashed, 67 h | 3,334,699 |
| 3 | **`zjhnye4j`** | https://wandb.ai/cath/voxelSmiles/runs/zjhnye4j | bubba-213-1, 2025-05-09T03:40:23Z, crashed, 92 h | 4,602,849 |

- The released step **3,960,000 falls inside run 3** (which spans 3.33 M → 4.60 M at epoch 0→1), so the
  released checkpoint is an **intermediate checkpoint of `zjhnye4j`**, not its final state.
- `zjhnye4j`'s `task_name` (`CombinedHiQBAggPropPoc2Mol`) and `ckpt_path` match the released
  `config.yaml` exactly. Its sibling **`7yzj4c06`** (`CombinedHiQBindHigherPropPoc2Mol`, ends 4,571,099)
  also resumed from `55vlvc7x` and also passes through 3.96 M — ruled out only by `task_name`.
- `crz11hbc` console log confirms dir `logs/vox2smilesZincAndPoc2MolOutputs/runs/2025-03-22_21-18-58`
  (the `_from_kaspian` suffix was added when the dir was copied to bubba).
- `55vlvc7x` and `zjhnye4j` have **no `output.log`** on the server (crashed bubba runs), so they're
  matched by timestamp + `task_name` + the `ckpt_path` chain, not by log. Timestamps convert as
  **BST = UTC+1** after 30 Mar 2025; March runs are UTC.

**This revises §4.** The two paper tables are *not* two unrelated models — they're most likely two
different checkpoints **of the same run `zjhnye4j`**: the May eval dir `bubba_zjhnye4j_2025-05-11_…`
evaluated a checkpoint taken ~May 11 while the run was still going (it ended ~May 12 23:30 UTC), whereas
the July eval used the released step-3,960,000 checkpoint. Still means Table 1 and Table 2 are not from
one checkpoint, but the gap is training steps within a run, not different models.

⚠️ **`crz11hbc` has `ckpt_path: None`** and was launched as `+experiment=train_vox2smiles_combined` (an
experiment config no longer in the repo). So the released decoder's traceable resume chain starts from
scratch at `crz11hbc`, and the separately-trained ZINC-only runs (`s7xnxqhu`, `lryunro2`, `t2gly4xp`) are
**not** in its lineage. The paper describes ZINC pretraining *then* fine-tuning on Poc2Mol grids; the
actual released model appears to have trained on the **mixture from the start** (that config used
`prob_poc2mol: 0.3`, so ~70% ZINC). Worth checking before repeating the paper's description.

### Re-training comparison baselines
Pull curves for `ljv96zyo` (Poc2Mol) and `zjhnye4j` (combined). Note both released checkpoints are
*intermediate*, so compare at matched `trainer/global_step`, not final values. Beware `_step` (W&B
logging counter) ≠ `trainer/global_step` — they differ by ~75× here.

---

## 5. Evaluation findings

### 5.1 The likelihood metric is dominated by a ligand-size nuisance term

`evaluate_combined_vox2smiles.py:352-394` writes one CSV per pocket under
`.../plixer_likelihood_scores/likelihood_scores/`. They store only `is_hit`, `likelihood` and the
*pocket* id — **no ligand identity** — but identity is fully recoverable because decoys are built as
`[s for s in df.smiles.values if s not in batch['smiles']]`, i.e. the test-split CSV in fixed order with
the pocket's own SMILES removed. Row 0 is the hit. Reconstruction verified exact (row counts match for
all 943 files including the 27 duplicate-SMILES systems; recomputed AUCs match the stored ones to 1e-16).
Script: see §7.

Variance decomposition of the 943×943 matrix:

```
pocket effect    6.4%
ligand effect   84.0%   <- intrinsic molecule likelihood, mostly SIZE
interaction      9.5%   <- the only part that can encode pocket specificity
```

Likelihood is mean per-token cross-entropy, so it tracks molecular size:
`corr(ligand mean likelihood, heavy-atom count) = -0.758`. For the ranking task this term is **pure
noise, not competing signal** — a pocket-blind baseline scoring by ligand mean alone gets AUC exactly
0.500. It just swamps the interaction term.

Removing it (z-score each ligand's column across pockets):

| Split | Raw (published) | Z-normalised |
|---|---|---|
| chrono | 0.67 | **0.854** |
| PLINDER | 0.58 | **0.753** |
| seq-sim | 0.61 | **0.782** |

Controls all pass (computed on the May run, where the full 943×943 matrix was reconstructed first):
the column view (fix ligand, rank pockets) independently gives 0.8493 vs the z-normalised 0.8497;
size-matched decoys (±10% heavy atoms) still 0.804; excluding same-protein sister pockets 0.8500;
permutation null 0.543; pocket-blind baseline exactly 0.500. No leakage — `forward` calls the same
`compute_smiles_metrics` on the same `predicted_ligand_voxels` used for decoys, and Poc2Mol only ever
sees protein voxels. Robust across both checkpoints (0.850 May / 0.854 July), so the conclusion does
not depend on which one you settle on.

**Implication:** on PLINDER this is 0.753 vs AutoDock Vina's 0.62. The paper's claim that Plixer
likelihoods are *"slightly less effective in ranking than docking scores"* is an artifact of the
normalisation. **Deployable prospectively** two ways: (a) score against a fixed background panel of
~100 reference pockets and z-score; (b) better — use the **likelihood ratio**
`log P(S | pocket) − log P(S | pocket-free reference)` with the ZINC-only decoder as reference. That's
the pointwise mutual information; the current metric is the numerator alone.

### 5.2 Data splits — protein-level clean, ligand-level leaky

Verified from the released checkpoints' own configs:
- **Poc2Mol**: `../hiqbind/parquet/{train,val,test}`. Test = chronological (2020+). Val = carved from the
  **pre-2020 pool by protein cluster**, *not* chronological — and cleanly cluster-disjoint from train.
- **Combined**: ZINC + Poc2Mol outputs over `../hiqbind/parquet/**train**` (correct split).
- **Vox2Smiles**: `../zinc20_parquet`, no protein split (ligand-only; deliberate).

| Check | Result |
|---|---|
| train ∩ val ∩ test system_ids | **0** everywhere |
| train/val protein clusters | **0 shared** — 10% cluster holdout worked |
| train ∩ test protein clusters | 191 — expected for a chronological split; this is why the strict subsets exist |
| 943 chrono / 107 PLINDER / 141 seq-sim eval systems | **all in parquet test, none in train** |

So no evaluation *pocket* was ever trained on. But **no ligand constraint was applied**:

- 21.3% chrono / 11.2% PLINDER / 14.2% seq-sim test ligands appear verbatim in Poc2Mol/combined **train**
  (paired with a different pocket).
- 24% chrono / 30% PLINDER / 37% seq-sim appear verbatim in **ZINC20**, seen by the decoder.

Measured impact (stratified by whether the true ligand was ever seen):

| Metric | Seen | Unseen | Published |
|---|---|---|---|
| chrono sim-enrichment | EF 10.29 | EF 6.08 | 7.58 |
| PLINDER sim-enrichment | EF 6.86 | EF 4.72 | 5.50 |
| chrono raw AUC | 0.758 | 0.618 | 0.67 |
| seq-sim raw AUC | 0.696 | **0.552** | 0.61 |
| chrono z-norm AUC | 0.880 | 0.839 | — |
| seq-sim z-norm AUC | 0.799 | 0.771 | — |

On genuinely unseen ligands the raw seq-sim AUC falls to near chance (0.552). The z-normalised metric is
**3–5× less sensitive** to this contamination — an independent argument for it.

Call the strict subsets **"protein-novel"**, not "non-redundant": they control protein redundancy only.
Report headline metrics stratified by ligand novelty rather than re-splitting and retraining.

⚠️ Leakage figures are **lower bounds** — matched as exact SMILES strings without re-canonicalising.

### 5.3 Known weaknesses of the published evaluation
- Vina correlates with heavy-atom count; report **ligand efficiency** (Vina/HAC) — Pocket2Mol beats Plixer
  on QED/LogP, so "better Vina" may partly mean "bigger molecules".
- The 0.3 Morgan-Tanimoto hit threshold is arbitrary and below the level that implies shared activity.
- Global ROC-AUC is the wrong VS summary — use **EF@1%, EF@5%, BEDROC (α=20)**.
- Fig. 3 shows a spike at similarity 1.0 (exact recoveries) — audit against memorisation.

---

## 6. Available but unused evaluation data

**On the node** — `../runs_n_poses/parquet/test/` (built 2026-08-06 by
`scripts/create_runs_n_poses_eval_set.py`): **1,836** drug-like systems / 1,800 PDB entries from
Runs N' Poses, all postdating the 2019-12-25 training cutoff, in `ParquetDataset` layout with
per-row NTAB novelty tiers (`max_tanimoto_to_train`, `novelty_tier`; **42.4%** in the hardest
`[0, 0.35)` tier). Replaces the underpowered 107-system PLINDER / 141-system seq-sim strict
splits. Rationale, two-stage filter cascade and caveats in `BENCHMARK_PLAN.md` §5a-i-b.
⚠️ Receptors are **unprotonated** — fine for `cons4`, invalid for `h5`/`hall6` (§3f).
Inspect the ligands with `scripts/adhoc_analysis/plot_rnp_ligand_sample.py` (paged contact sheets;
200 molecules in one image is not legible). Doing exactly that is what caught the first build
shipping ~19% nucleotides, glycans and fragments past the CCD blocklist — **look at the molecules
before trusting a chemotype filter.**

The rest are **local-only** (`/mnt/disk2/VoxelDiffOuter/...`, not copied to the node) and would
replace the arbitrary similarity proxy with real actives/inactives:

- `../LIT-PCBA_AVE_UNBIASED` — 15 targets, AVE-debiased, dose-response actives/inactives. **Best primary
  choice.** A started script exists: `evaluations/evaluate_combi_model_lit_pcba.py`.
- `../DUDe_binding_and_decoys` — 102 targets, property-matched decoys. Gap vs LIT-PCBA is diagnostic of
  property shortcuts.
- `../BindingDB/BindingDB_All.tsv` — replaces "the one PDB ligand" with *all* known actives per target.

Also worth adding: **PoseBusters** on docked generated molecules; **Boltz-2** affinity as a much better
oracle than Vina (`.boltz` already present on the node — but it's a model, not ground truth).

---

## 7. Analysis scripts

Now in the repo under `scripts/adhoc_analysis/` (prefix `paper_audit_`). Paths inside them are
absolute to the **local** `/mnt/disk2/VoxelDiffOuter/...` machine — they need repointing to
`~/plixer_outer/...` to run on the node, and they read `evaluation_results/`, which is gitignored
and currently **local-only** (48 GB, not copied to the node).

| Script | Does |
|---|---|
| `paper_audit_rebuild_matrix.py` | reconstructs the 943×943 pocket×ligand likelihood matrix (§5.1) |
| `paper_audit_analyse_matrix.py` | variance decomposition, z-normalisation, bootstrap CIs |
| `paper_audit_validate.py` | cross-check vs stored AUCs, size control, permutation null |
| `paper_audit_paper_run.py` | reproduces Table 2 from the July run + z-normalised values |
| `paper_audit_split_audit.py` | train/val/test overlap + ZINC leakage scan |
| `paper_audit_leak_impact.py` | stratifies metrics by ligand-seen-in-training |
| `paper_audit_wandb_runs.py` / `_lineage.py` / `_confirm.py` | W&B run inventory + lineage (§4b) |

## 7b. Provenance system (added 2026-08-06, uncommitted at time of writing)

`src/utils/provenance.py` + `ProvenanceCallback`, wired into `src/train.py` and mirrored into W&B
hparams by `logging_utils.py`; `scripts/describe_checkpoint.py` reads it back. Records git commit /
branch / dirty state, resolved config, parent checkpoints and hostname, **embedded into every
checkpoint**. This exists precisely because §4b was so painful to reconstruct — future runs should be
identifiable from a stray `.ckpt` alone. Use it for all re-training runs.

## 7c. Manuscript

`plixer-manuscript/` is a **separate git repo** nested in the working dir
(`git@github.com:JudeWells/plixer-manuscript.git`, local branch `phd_thesis_ablation` = origin/main
@ d397015). It is not tracked by the plixer repo and not in `.gitignore` — adding it would create an
embedded-repo warning. Leave it untracked or gitignore it.

## 8. Open items

1. Rerun eval against `checkpoints/` to confirm the released weights are the July model
   (§4b suggests both tables trace to run `zjhnye4j` at different steps — verify).
2. Regenerate Table 1 + Table 2 from a **single** checkpoint.
3. Decide what replaces Vina (needs re-docking from scratch either way).
4. Implement the likelihood-ratio scorer; check it reproduces the z-normalised numbers.
5. Redo leakage matching with canonical SMILES + Murcko scaffolds for a true figure.
6. Test DDP for the combined stage before any multi-GPU run.
6b. **Run the §3f protonation-leakage diagnostics before trusting any `h5`/`hall6` result.**
   Start with the H-bond geometry test and the HIS tautomer skew — both are cheap and need no
   retraining. Until then, sweep 2's hydrogen arms are uninterpretable.
7. ~~Port `inference/` and `evaluations/` off `UnifiedVoxelGrid`.~~ **Done 2026-08-06.** Rather than
   rewriting each caller, `voxelize_complex` / `voxelize_molecule` (`molecule_utils.py`) and
   `voxelize_protein` (`utils/utils.py`, the `generate_smiles_from_pdb.py` entry point) were rerouted
   through the batched voxeliser, so every caller inherits the corrected numerics unchanged.
   `tests/test_eval_path_consistency.py` asserts training and evaluation produce **bit-identical**
   grids — if they drift, every reported number is measured on data the model never saw.
   No live `UnifiedVoxelGrid(` call sites remain outside `voxelizer.py` itself.
   Also removed: a "fallback voxelisation" in `evaluate_vox2smiles.py` that could never have run
   (it called a non-existent `voxelize_ligand`, and double-applied the augmentations).
8. The three §3b bugs mean released checkpoints and newly trained ones are not comparable on equal
   footing. Any table mixing them needs a caveat, or re-evaluation of the released weights under the
   old encoding (which the config flags still permit).

---

## 9. Experiment 1 — protein channels into the SMILES decoder (in progress, 2026-08-06)

Feed the 4 protein channels into Vox2Smiles alongside the 9 ligand channels, so the decoder sees pocket
context rather than only Poc2Mol's predicted ligand density. ZINC ligand-only pretraining continues to
work by masking the protein channels; masking is also applied with some probability during combined
training so the model does not become dependent on them.

Decisions taken (2026-08-06):

- **Poc2Mol is trained once and shared frozen by both arms.** Its architecture is unchanged by this
  intervention (protein voxels → ligand voxels), so sharing removes a confound and halves the compute.
- **Protein channels fixed to C/O/N/S** (§3b bug 1), 4 channels, not widened. **§3f retroactively
  supports this**: adding a hydrogen channel would have imported HiQBind's ligand-conditioned
  protonation into the decoder as well, on top of Poc2Mol.
- **Matched-budget, seed-replicated comparison**: both arms from scratch, identical data / steps /
  effective batch / LR schedule, 2–3 seeds, so the delta gets an error bar. Only channel count and
  protein masking differ.
- Free extra diagnostic: evaluate the intervention model with protein channels masked at inference. If
  that recovers baseline performance, the model is genuinely using the protein signal.

Baseline for this experiment is a **re-implementation**, not the published model — the §3b fixes make
them different pipelines. That is a stated limitation, not an accident.

### Configs (all four run under DDP as of 2026-08-06)

| | stage 1 (ZINC pretrain) | stage 2 (combined finetune) |
|---|---|---|
| baseline | `exp1_zinc_baseline` (9 ch) | `exp1_combined_baseline` (9 ch) |
| protein | `exp1_zinc_protein` (14 ch) | `exp1_combined_protein` (14 ch, mask 0.25) |

14 = 9 ligand + 4 protein + **1 constant "protein present" flag plane**. The flag matters because empty
protein channels are not self-identifying — a sparse pocket and a masked one both read as near-zero.
Since the ViT's first op is a Conv3d over the patch, a constant plane adds a constant vector to every
patch embedding, i.e. it is exactly a learned "protein absent" embedding for the price of one channel.
Assembly and masking live in `src/data/common/protein_channels.py`; `tests/test_protein_injection.py`
checks that the baseline arm is bit-identical to not having the feature, that ZINC reads as absent, and
that a masked pocket sample is indistinguishable from a ligand-only one.

Stage-1 weights are **not** interchangeable between arms — the patch-embedding conv has a different
input width. Loading the wrong one fails there.

### Two config traps found while wiring this up

1. **`trainer.accumulate_grad_batches` in an experiment config is ignored.** `src/train.py` overwrites it
   from `data.config.target_samples_per_batch / (batch_size × world_size)`. `Vox2SmilesDataConfig` had no
   `target_samples_per_batch` field, so it fell back to `batch_size` → accumulation 1, meaning the
   `accumulate_grad_batches: 16` in `train_vox2smiles_combined_hiqbind.yaml` **never took effect** and the
   published combined run trained at an effective batch of 64, not 1024. The field now exists; set it
   explicitly. `train.py` logs the resolved effective batch and warns when it does not divide evenly.
2. **The world size now divides out of the accumulation.** Without that, moving 1 → 8 GPUs silently
   multiplies the effective batch by 8 and no cross-world-size comparison is valid.

### DDP verification (2026-08-06)

Matched at 12 optimiser steps, effective batch 512: **1 GPU `val/loss` 1.66205 vs 8 GPU 1.66056** (0.09%).
Exact equality is not achievable — DDP shards the data, so the two runs consume different samples.

Fixes that DDP required, beyond the data pipeline: the ViT pooler was instantiated but never read by
`VisionEncoderDecoderModel`, so its weights got no gradient and DDP hard-errors — now
`add_pooling_layer=False` (a no-op mathematically, it was dead weight). `wandb.log` was called on all
ranks from `visualize_smiles` and `visualise_batch`. Validation metrics lacked `sync_dist=True`, so
`val/loss` reflected rank zero's shard only — which would have mis-selected checkpoints.

### Sizing note

HiQBind is only **9,872 clusters**, so at effective batch 512 an epoch is **19 optimiser steps**.
`limit_train_batches` above ~19/rank does not bind on 8 GPUs, and per-epoch overhead dominates short
runs. Choose the effective batch deliberately for the Poc2Mol retrain rather than inheriting 512.

⚠️ **`max_poc2mol_loss` cannot simply be recalibrated** — see §3c. The absolute threshold gates on
ligand composition rather than reconstruction quality, because the per-sample loss floor varies
0.473–0.802 with how many ligand channels are occupied. Either remove the filter (leading option) or
make it relative to each sample's floor. For reference, the released `epoch_173.ckpt` scores 0.816 on
val under the old encoding and 0.968 under the corrected one; the inherited 0.79 rejects everything.

## 9b. The optimiser noise floor — run `ciqwntcy` (measured 2026-08-07)

`s1_protein_resume` ([ciqwntcy](https://wandb.ai/cath/voxelSmiles/runs/ciqwntcy)) resumed from a
weights-only checkpoint (val/loss 0.1866) at lr 1e-4 behind a 4000-step warmup. The warmup ramp
accidentally produced a clean sweep of the LR/loss relationship at fixed model state:

| lr | val/loss |
|---|---|
| 1.18e-05 | 0.1764 |
| **2.43e-05** | **0.1758**  ← minimum |
| 3.68e-05 | 0.1771 |
| 4.93e-05 | 0.1805 |
| 6.18e-05 | 0.1872 |
| 8.68e-05 | 0.1918 |
| 1.00e-04 | 0.1992 |
| 1e-4 constant thereafter | 0.194–0.206 (mean 0.198, n=143) |

val/loss is **monotone increasing in LR**. This is the optimiser noise floor: Adam does not converge
to a minimum, it equilibrates in a ball around it whose radius scales with the learning rate, so the
stationary loss is ≈ `L* + c·lr`. Fitting that to the two ends extrapolates to **L\* ≈ 0.169 at zero
LR**, and the minimum sampled was at the *lowest* LR tried — no floor had been reached by 2.4e-5.

🚨 **THE HEADLINE CONCLUSION OF THIS SECTION IS FALSIFIED — see §9d.** The production runs reach
`val/loss` **0.019–0.033 at 22k steps at lr 3e-4**, i.e. a 3× *higher* LR reaching a 10× *lower* loss
than this section claims is the floor. The local LR/loss monotonicity below is real, but it was
measured around a bad optimum: ~0.17 was that particular model's capability, not a universal floor,
and the `L* ≈ 0.169` extrapolation is wrong by an order of magnitude. Do not cite consequence 1.

Three consequences:

1. ~~**The model was LR-noise-limited at 154k steps, not data- or capacity-limited.**~~ **WRONG**
   (§9d). It was limited by whatever the Aug-7 code changes fixed. The decay phase
   is worth ~0.02–0.03 val/loss on its own, independent of any further learning.
2. **Mid-training val/loss at full LR systematically understates every model.** Comparing the
   `maxagg`/`sumagg` arms during the stable phase measures their noise floors, not their potential.
   The arm comparison must be made **after decay**, or at matched LR.
3. **Do not pick an LR from early curves.** Higher LR wins early *and* has a higher noise floor, so
   the ranking can invert after decay. 5e-4 led the from-scratch sweep at 4k steps; that does not
   make it the best production LR.

It also settles the resume question: at matched LR with a 4000-step warmup there was **no spike at
all** — val went 0.195 → 0.176 immediately. The earlier catastrophic 0.195 → 0.52 was the
*combination* of missing Adam state, a 2–5× LR increase, and a 500-step warmup, not resuming per se.

## 9c. Stage-1 PRODUCTION runs (launched 2026-08-07)

`scripts/run_exp1_stage1_production.sh`, logs in `logs/exp1_stage1_prod/20260807_182209/`.
Two arms differing **only** in voxel aggregation (`voxel_radius_scale` deliberately left at 1.0 in
both, so aggregation is the single variable):

| arm | `voxel_aggregation` | GPUs | W&B |
|---|---|---|---|
| `maxagg` | `max` (current default) | 0–3 | [cpethcmn](https://wandb.ai/cath/voxelSmiles/runs/cpethcmn) |
| `sumagg` | `sum` (additive) | 4–7 | [00b1tt98](https://wandb.ai/cath/voxelSmiles/runs/00b1tt98) |

Schedule (identical across arms): lr 3e-4, warmup-stable-decay 2k / 298k / 200k, **`min_lr_ratio`
0.03** (→ 9e-6), `max_steps` 500000, batch 128/GPU × 4 GPUs × accumulation 1 = **effective batch
512**, `val_check_interval` 5000, `num_channels` 14. `save_weights_only=False` this time — the
weights-only checkpoint is what made the earlier resume lossy (§9b).

`min_lr_ratio` is 0.03 rather than 0.1 because §9b found the loss still falling at the lowest LR
sampled; 0.03× lands at 9e-6, below anything that run measured.

Throughput **1.6 it/s → ~87 h (3.6 days) per arm**, run concurrently. Loss 4.32 → ~0.55 by step 1000
(4.32 ≈ ln 77, the uniform-prior value over the 77-token vocab, i.e. init is sane).

### Launch bug: Hydra silently killed the first attempt

The first launch (`20260807_180818`) died **within one second**, leaving all 8 GPUs idle, and was
reported as running because the launcher only backgrounded the process and never checked it:

```
ConfigCompositionException: Could not override 'data.config.voxel_aggregation'.
To append to your config use +data.config.voxel_aggregation=max
```

`voxel_aggregation` and `voxel_radius_scale` exist as `Vox2SmilesDataConfig` dataclass fields but
were not declared in `configs/data/vox2smiles_zinc_data.yaml`, and Hydra refuses a bare override for
a key absent from the config node. **Both fields are now declared explicitly in that YAML** so plain
overrides work — preferred over sprinkling `+` at call sites, which fails the other way once the key
does exist. Two guards added:

* `run_exp1_stage1_production.sh` now verifies both PIDs are alive 180 s after launch and exits
  non-zero with the log tail if not.
* Validate override lists with `python src/train.py --cfg job --resolve <overrides>` before
  launching anything long. Hydra applies overrides in order and reports only the *first* failure, so
  a successful dry run is the only proof the whole list is good.

## 9d. The stage-1 jump is REAL — verified three ways (2026-08-07)

The production runs looked too good (`val/loss` 0.0194–0.0327 and token accuracy 0.984–0.990 at
**22k** steps, against the old stage-1 runs' 0.187–0.189 / 0.921 at **128–142k** steps). It is not a
metric bug. Three independent checks, all on `maxagg`'s `last.ckpt` under the live val pipeline:

| check | result | reads as |
|---|---|---|
| teacher-forced token accuracy | 0.9748 | reproduces the W&B number |
| **free-running greedy exact match** | **0.2767** (83/300) | 0.9748⁴⁰ ≈ 0.36; consistent, and impossible without real conditioning |
| **zeroed-voxel ablation** | acc **0.4413**, loss **2.2865** | removing the input costs 53 accuracy points and 45× the loss |

The unconditional SMILES prior scores 0.44, so the model is genuinely reading the voxels, and it
reconstructs the *exact* target molecule 28% of the time from free-running generation. Generations
inspected by eye are chemically sensible and stereochemically detailed.

Ruled out as explanations:

* **Not a padding-inflated denominator.** `validation_step` maps pad → −100 and calls
  `accuracy_from_outputs(..., ignore_index=-100)`; the two agree, so pad is excluded.
* **Not a teacher-forcing/off-by-one leak.** `start_ix=1` slices logits and labels identically,
  which is the correct alignment because HF builds `decoder_input_ids` via `shift_tokens_right`
  internally. And a leak there would not survive free-running generation, which it does.
* **Not the data.** Every data-affecting config key is byte-identical across all four runs —
  `include_hydrogens: false`, `max_smiles_len: 200`, `index_train.csv`/`index_val.csv`,
  rotation/translation, box, channels. Val set is the same 104,029 molecules.
* **Not the voxeliser formula.** `batched.py:305` multiplies `1/d`, giving `1 − exp(−(vdw·scale/d)¹²)`
  — correct. (⚠️ `pytest` is **not installed in the venv**, so `tests/test_batched_voxelizer.py` has
  not actually been run since those edits. Install it and run the suite.)

### What is NOT explained, and why it cannot be bisected

The only differences from the old runs are lr 1e-4 → 3e-4, effective batch 256 → 512, and **code
edited on 2026-08-07**: `src/models/vox2smiles.py` (06:52, the §3i token fix) and
`batched.py` / `config.py` / `poc2mol/collate.py` / `vox2smiles/data_module.py` /
`poc2mol_inference.py` (18:02–18:03, the aggregation/`radius_scale` work — 19 minutes before
launch). lr and batch cannot plausibly buy 10× loss, so one of those edits fixed something real.

**Which one is unknowable: the entire working tree is uncommitted, so there is no history to bisect
and the old code no longer exists on disk.** The old runs' checkpoint directories are also gone.

⚠️ **Commit the working tree.** 19 files / ~774 insertions are uncommitted, and the provenance
system (§7b) records a commit hash that is meaningless against a dirty tree — so right now the
production runs are *not* reproducible, which is the exact failure §7b was built to prevent. This is
also why §9b's conclusion survived unchallenged for a day.

Leading hypothesis (untested): a defect in the pre-18:02 batched voxeliser degraded the ligand grid
— the box-reach filter is the prime suspect, since train uses `random_translation: 6.0` and a filter
that pruned slightly too aggressively would silently drop ligand atoms near the box edge. The old
model was still conditioned (0.92 ≫ the 0.44 prior), just working from a corrupted grid. Testable by
reconstructing the old voxeliser from the diff against HEAD and comparing grids on fixed input.

### Whole-SMILES exact match — the metric worth quoting (measured 2026-08-08)

Token accuracy saturates and is hard to read; free-running exact match is the honest number.
Greedy decode, 2,400 held-out ZINC molecules per arm, both at **global_step 66,912** (13% of the
500k run, still in the stable phase at full LR — the 200k decay has not started):

| | maxagg | sumagg |
|---|---|---|
| teacher-forced token acc | 0.9922 | 0.9931 |
| validity | 0.9992 | 0.9967 |
| **exact match (string)** | **62.58%** | **64.62%** |
| **exact match (molecule, canonical)** | **62.83%** | **64.88%** |

* **String ≈ canonical** (+0.25 pp, ~6 molecules per 2,400). The model reproduces RDKit's canonical
  form rather than an equivalent rewrite, so cheap string comparison does not under-count.
* Consistent with token accuracy: 0.9922⁴⁰ ≈ 0.73 vs 0.626 observed, lower because token errors
  cluster within a molecule.
* Trajectory: **27.7% @22k → 62.6% @67k**. For contrast the earlier per-token analysis measured
  4.7% exact at token acc 0.923.
* ⚠️ **The +2.0 pp sumagg lead is ~1.5σ** (unpaired SE on the difference ≈1.4 pp at n=2400) and runs
  the other way on validity. Not a result yet; compare after decay (§9b).
* ⚠️ This is the **stage-1 task** — ground-truth ligand voxels, no pocket. Do not carry 63% forward
  as an expectation for stage 3, which conditions on Poc2Mol's predicted density.

Script: `/tmp/.../scratchpad/exact.py` pattern — instantiate the datamodule via Hydra, call
`dm.on_after_batch_transfer` to build voxels, then `model.model.generate(...)`. Worth promoting into
`evaluations/` as the standard stage-1 readout.

### `val/exact_match` is now logged during training (added 2026-08-08)

`calculate_exact_match` (`src/utils/metrics.py`) + `val/exact_match` / `val/poc2mol_output/exact_match`
in `VoxToSmilesModel.validation_step`. **Computed from the SAME `generate_smiles` call that already
feeds validity, so it costs no extra decode.** Comparison is RDKit-canonical, not string: benzene
written `C1=CC=CC=C1` vs `c1ccccc1` is a match, which string comparison would miss. Unparseable
generations count as misses, never errors.

This is the metric to watch from here — token accuracy (>0.99) and validity (>0.999) have both
saturated and no longer discriminate; exact match was 63% and still climbing at the same point.

⚠️ **Sample size.** It reuses validity's sample: `pixel_values[:30]` from batch 0 only, per rank
(§3i), so ~120 molecules on 4 ranks → SE ≈ 4.4 pp at p≈0.63. Fine as a trend line, too noisy to
separate the arms. Raise `model.n_samples_for_validity_testing` to tighten it (both metrics
benefit), or use the standalone script below for a real number.

⚠️ **The running production runs will NOT show it** — Python imported the module at launch, so the
change only takes effect on a new run or a resume.

Standalone: `evaluations/evaluate_stage1_exact_match.py --checkpoint <ckpt> --voxel_aggregation
{max,sum} [--n_batches N] [--zero_voxels] [--output x.json]`. Reports teacher-forced token accuracy,
validity, and exact match both string and molecule-level. `--zero_voxels` runs the §9d conditioning
ablation. ⚠️ `--voxel_aggregation` **must match what the checkpoint was trained with**, or the input
distribution differs from training and the numbers are meaningless.

Also fixed while here: `calculate_validity` and `is_valid_smiles` now scope RDKit's parse-error
logging with `rdBase.BlockLogs`. An untrained decoder was emitting a multi-line RDKit error per
sample per validation — 120 lines per validation step in a smoke test, now 0.

## 10. Stage 2/3 de-risking audit (2026-08-09)

Done before committing a second node to the decoder fine-tune. Three defects and one
incompatibility, in order of how much they cost if missed.

### 10a. BUG FIXED — stage 2 zeroed the pocket for every training sample

`poc2mol_inference.py` masked the protein channels on `select` (rows using Poc2Mol's
*prediction*) instead of `needs` (rows that *have* a pocket):

```python
protein = torch.where(select, protein, torch.zeros_like(protein))   # WRONG
```

Stage 2 sets `predicted_ligand_probability: 0.0`, so `fraction = 0` → `use_pred` all False →
`select` all False → **the pocket was zeroed for 100% of stage-2 training samples**, i.e. the
stage whose entire purpose is "learn to read the protein channels" would have trained with them
blank. Validation uses `fraction = 1.0`, so validation *did* show the pocket — a train/serve
mismatch that `val/loss` could not have revealed. Invisible in stage 3 at the end of the ramp
(`fraction = 1` makes the two masks identical), which is why a stage-3 smoke test missed it.
Now masks on `needs`.

### 10b. ⚠️ `sumagg` is INCOMPATIBLE with the frozen Poc2Mol — blocks the leading arm

Poc2Mol (`cons4_drop`, trained 2026-08-06) predates `voxel_aggregation` entirely: its
`resolved_config.yaml` has no such key, so it trained on **max**-aggregated grids for both its
protein input and its ligand target. Two independent breakages if stage 2/3 runs with `sum`:

1. **Input.** `Poc2MolInferenceBuilder._bind` builds ONE `BatchedVoxelizer` from `data.config`
   with `aggregation=voxel_config.get("voxel_aggregation", "max")`, and that single grid is split
   into the protein channels fed to the frozen Poc2Mol. Under `sum` those run 0–~5.6 instead of
   0–1: out of distribution for a model that has only ever seen `max`.
2. **Output.** `predicted = torch.sigmoid(predicted_logits)` is in [0, 1] by construction, but a
   `sum`-pretrained decoder expects 0–~5.6. In stage 3 the decoder would receive *true* ligand
   grids at 0–5.6 and *predicted* ones at 0–1 — two different input scales mixed inside one batch,
   getting worse as the ramp advances.

`max` is coherent end to end (true 0–1, predicted 0–1). **So the arm currently winning on exact
match cannot proceed past stage 1 as things stand.** Fix is cheap: retrain Poc2Mol with
`data.config.voxel_aggregation=sum`. `cons4_drop` reached epoch 307 in **3 h 06 m**, so ~3 h on one
GPU, and it doubles as a test of whether `sum` helps Poc2Mol itself.

### 10c. Stage 2 → stage 3 resume is weights-only

`exp1_s2_*.yaml` and `exp1_s3_*.yaml` both set `save_weights_only: True`, and stage 3 resumes from
stage 2. §9b showed a weights-only resume (no Adam state) is exactly what produced the lossy
0.195 → 0.52 excursion. Set `False` for stage 2 at minimum.

### 10d. Still open before a multi-GPU combined run

* **DDP untested for the combined stage** (open item 6). `CombinedDataset` mixes ZINC and complexes
  by sampling probability; under DDP each rank draws independently. Needs the same matched
  short-run check that validated stage 1 (1 GPU vs 4 GPU `val/loss`).
* **Sizing.** HiQBind is 9,872 clusters. At the configured batch 64 × 4 = 256 with
  `prob_poc2mol: 0.5`, an epoch is ~20–40 optimiser steps, against `num_stable_steps: 200000`.
  Choose the schedule from the data, not by inheriting stage 1's.
* **No matched baseline for experiment 1.** A 9-channel stage-1 baseline checkpoint exists
  (`logs/exp1_s1_baseline/.../last.ckpt`, step 135,821) but it is from the **pre-2026-08-07 code
  era** that plateaued at `val/loss` 0.19 (§9d), and is weights-only. It is not a valid control for
  the new 14-channel runs. Either run a 9-ch arm under the current code (~87 h) or rest the claim
  on the `--mask_protein` within-model ablation, and say which.

### 10e. Correction to §7b

Provenance does **not** embed the resolved config. The keys actually present are `env, epoch, git,
global_step, parent_checkpoints, run_dir, run_id, saved_at, schema_version, seed, tags, task_name`.
The resolved config is written separately to `<run_dir>/resolved_config.yaml`, which is fine but is
not carried inside a stray `.ckpt` — so a checkpoint alone does not tell you its voxelisation
settings, which is exactly what 10b turned on.

## 11. Decoder fine-tune on ZINC + Poc2Mol outputs (launched 2026-08-10, nebius2)

Second node `ssh nebius2` (8x H100, 128 cores, 1.5 TB). Provisioned from the primary node by
rsync: uv-managed CPython 3.11 (the venv symlinks to it and nebius2 had only 3.12), venvPlixer,
repo, hiqbind + zinc parquet, the shared Poc2Mol, and the stage-1 maxagg decoder as
`checkpoints/s1_maxagg_last.ckpt`. Verified there: torch 2.3.1+cu121, 8 GPUs, docktgrid
`DTYPE = torch.bfloat16` patch intact.

**Run `s3_maxagg`** — [im1j69zk](https://wandb.ai/cath/voxelSmiles/runs/im1j69zk), GPUs 0-3,
3.6 it/s. From stage-1 maxagg weights, `prob_poc2mol 0.5`, predicted-density ramp over steps
2k-15k, lr 5e-5, warmup 1000 / stable 39000 / decay 20000 to 0.03x, max_steps 60000, effective
batch 256, `quality_filter: none`.

Ramping inside one run rather than launching stage 2 then stage 3 separately: steps 0-2k are the
stage-2 regime (true voxels, real pocket), and the ramp reaches the deployed regime by 15k.
Validation always uses the prediction, so `val/poc2mol/*` is the deployed metric from step 0.

### `init_weights_from` — new, and necessary

`ckpt_path` goes to `trainer.fit` and restores global_step, optimizer AND scheduler. Starting a
curriculum stage that way would begin at stage 1's ~100k steps, past its own warmup and possibly
past `max_steps`. `init_weights_from` (`configs/train.yaml`, handled in `src/train.py`) loads the
tensors only and starts at step 0 with a fresh schedule. It records the parent in provenance and
warns on any non-metric state_dict mismatch — which is how a 9ch/14ch arm mix-up would surface,
since the patch-embedding conv width differs.

### Validation is now reported three ways

`val/{zinc,poc2mol,combined}/{loss,accuracy,validity,exact_match,tanimoto}`. `combined` is the
pooled value, not a third dataset; `val/loss` is an alias of `combined/loss` so existing
checkpoint monitoring still works. Splits come from `Vox2SmilesDataModule.val_dataset_kinds`,
classified by dataset class, so reordering val datasets cannot silently relabel a curve.

A **ZINC val dataset was added to the combined config** — it had only pocket datasets, so
nothing could have detected the decoder trading away its pretraining ability for pocket
performance. Same held-out 104,029 molecules as stage 1, deterministic.

All logging happens in `on_validation_epoch_end`, not in the step: the pooled `combined` key is
fed from several dataloaders and Lightning rejects one key logged from several dataloader_idx.

### BUG FIXED — `calculate_average_similarity` was the wrong metric

It takes, per generation, the **max** Tanimoto over the ENTIRE reference set, so a generated
benzene matches any benzene in the batch and scores 1.0. Measured 1.0 on a triple where only one
pair was identical. It is a set-coverage measure, inflates with batch size, and is blind to
whether the model produced the right molecule for the right pocket. Replaced for this purpose by
`calculate_paired_similarity` (index-aligned, invalid generations score 0 rather than being
dropped, Morgan r=2/2048 as in the paper). The old function is left in place for
`calculate_metrics`, which uses it for a genuinely set-level question.

### Likelihood ranking, raw and z-normalised

`src/utils/likelihood_eval.py`, accumulated over the val epoch (z-scoring a ligand needs it
scored against many pockets, so it cannot be done per batch) and all-gathered across ranks.
Logs `val/likelihood_auc_{raw,znorm,pocket_blind}`. **Quote the z-normalised one** — §5.1.
`pocket_blind` is a live control that must sit at ~0.5; on synthetic data with a large
per-ligand offset the module gives raw 0.578 / znorm 0.980 / blind exactly 0.500.
Positives are matched by SMILES identity, not index, and non-parsing candidates are excluded —
the shipped 105-entry PLINDER decoy panel has one corrupt entry
(`"1Cc1ccccc1O=C(...)"`, two SMILES concatenated), 104/105 parse.

### Zero-shot baseline, measured before any fine-tuning

From the smoke run (stage-1 maxagg weights, no fine-tuning), which is the "before" row:

| split | loss | exact match |
|---|---|---|
| zinc | 0.0194 | 0.625 |
| poc2mol | 1.480 | **0.000** |

`val/likelihood_auc_znorm` **0.637** even zero-shot. So the stage-1 decoder cannot reconstruct
anything from Poc2Mol density (0% exact match) but its likelihoods already rank the true ligand
above decoys — the fine-tune has clear headroom, and the zinc row is the number to watch for
regression.

⚠️ `max_poc2mol_loss` is off (`quality_filter: none`) and should stay off: §3c showed the
absolute threshold gates on ligand *composition* rather than reconstruction quality, since the
per-sample loss floor swings 0.473-0.802 with how many channels the ligand occupies.

### Leave-one-out z-normalisation is provably pointless (checked 2026-08-10)

The column statistics include the entry being normalised, which looks like it should bias the
metric. It does shift the z-*values* (mean |Δz| ≈ 0.03 at P=104) but has **exactly zero** effect
on the reported AUROC. Substituting the closed-form LOO moments, the column mean and sd cancel:

```
z_loo = u · sqrt( P / ((P−1)(1 − u²/(P−1))) )        u = the in-sample z
```

— a strictly increasing function of `u` alone, applied identically to every entry. Ranking within
a pocket is therefore unchanged, so **AUROC, EF@k and BEDROC are all invariant**. Verified
empirically at P = 10/40/104/300: within-row Spearman exactly 1.000000000000, AUCs equal to every
digit. The LOO implementation was written, confirmed against a naive per-row recomputation to
1.6e-14, then **deleted** as dead weight.

What this does NOT settle: normalising against an **independent background panel** of pockets is a
different computation (separate mean/sd estimates, not a rescaling), which *can* reorder a row.
That is the prospective-deployment question from §5.1 and is still open — the experiment is to hold
out ~50 pockets as a fixed panel and z-score the rest against it.

Only non-rank uses of the z-values would notice the LOO distinction: absolute thresholds,
calibration, or comparing confidence across pockets.

### Batch-size arm (launched 2026-08-10)

Second arm on the free GPUs 4-7 to test whether a larger batch stabilises the fine-tune.

| arm | batch/GPU | effective | GPU mem | it/s | W&B |
|---|---|---|---|---|---|
| `s3_maxagg` | 64 | 256 | 36.8 GB | 3.6 | [im1j69zk](https://wandb.ai/cath/voxelSmiles/runs/im1j69zk) |
| `s3_maxagg_bs600` | **150** | **600** | **78.9 GB** | 1.34 | [y4ce35kc](https://wandb.ai/cath/voxelSmiles/runs/y4ce35kc) |

**lr is deliberately NOT scaled** (5e-5 in both). Raising both at once would confound "does a bigger
batch stabilise training" with "does a different lr stabilise training"; a larger batch at fixed lr
is exactly the reduced-gradient-noise condition the question is about.

The schedule is identical in STEPS (ramp 2k-15k, val every 2k, max 60k) so the curves overlay
directly on the step axis, which is the readable comparison for smoothness. ⚠️ Consequence: at
60k steps the bs600 arm consumes **2.3x the samples**. Fine for judging stability, NOT a
matched-compute comparison of final quality — for that, compare at matched `train/n_samples`.

Memory scales linearly and predictably: 36.8 GB at 64 and 68.3 GB at 128 give ~0.49 GB/sample plus
~5.3 GB fixed, which predicted 79 GB at 150 against an actual 78.9 GB. That leaves only ~2.6 GB
headroom on an 80 GB card, so **150 is the practical ceiling here** — 160 would not fit. An
intermediate bs512 (128/GPU) arm was launched and killed after 5 minutes when the ceiling was
pushed to 150.

### Metric rename + per-source training LM loss (2026-08-10)

`train/poc2mol_loss` -> **`train/poc2mol_voxel_loss`**. It was never a language-modelling loss: it
is the frozen upstream's per-sample BCEDice between its predicted ligand density and the true grid
(`poc2mol_loss_per_sample`), logged only for rows actually using the prediction. Sitting next to
`val/poc2mol/loss`, which IS a cross-entropy, the old name was actively misleading. Scale settles
it: the voxel loss runs ~0.78-0.87 (Poc2Mol's own val/loss is 0.8039 against a 0.6403 floor) while
the decoder's CE is ~0.05 — a factor of ~16.

New: **`train/{zinc,poc2mol_true,poc2mol_pred}/lm_loss`** — the decoder's cross-entropy on the
TRAINING data, split by source. This is what distinguishes the two readings of a rising
`val/poc2mol/loss`: if `train/poc2mol_pred/lm_loss` FALLS while val rises, it is memorisation of
the 9,872 HiQBind clusters; if both rise, the pocket task itself is regressing. Three groups
because during the ramp a complex row may carry either its true ligand voxels or Poc2Mol's
prediction; `poc2mol_pred` is the one directly comparable to `val/poc2mol/loss`, which always uses
the prediction.

Needed `has_pocket` (the per-sample `needs_poc2mol` flag) kept on the batch rather than popped —
`poc2mol_loss > 0` alone cannot separate a ligand-only row from a complex row using true voxels.

Verified: unit test on a controlled batch gives the right 2/2/2 split with exhaustive, disjoint
groups and 0.0 loss for rows made near-perfect; end-to-end, a real batch yields
`has_pocket=[1,1,1,0,...]`, voxel losses 0.76-0.88 on pocket rows and -1 elsewhere.

⚠️ Applies to NEW runs only — the two in-flight jobs imported the old module at launch.

## 12. What the density actually carries — ceiling, orthogonality, protein ablation (2026-08-10)

All measured on the 104-pocket / 105-candidate PLINDER panel with Poc2Mol's PREDICTED density,
from `s3_maxagg` step 4000 (`val/loss` 0.0106, the step whose z-norm AUC was logged as 0.7449).
Scripts: `scripts/adhoc_analysis/poc2mol_density_ceiling.py`, `decoder_vs_composition.py`.

### 12a. A parameter-free composition readout matches the decoder

Score each candidate by how well its per-channel heavy-atom counts match the summed per-channel
occupancy of the predicted density (calibrated to atom-count units, pose-free so decoys need no
3D structure):

| readout | raw AUC | z-norm AUC |
|---|---|---|
| size only | 0.652 | 0.668 |
| composition only (cosine) | 0.606 | 0.638 |
| **size + composition** | **0.689** | **0.761** |
| decoder | 0.547 | 0.722 |
| **ensemble (50/50, row-standardised)** | — | **0.785** |

⚠️ An earlier version of this test reported "size+composition" and "composition only" as
identical. They were: dividing a vector by its sum is a scalar rescale and cosine is already
scale-invariant, so the "size-sensitive" variant was a no-op. Size sensitivity needs a distance,
not a cosine.

### 12b. Poc2Mol's rare-element channels carry no information

| channel | r(predicted mass, true count) | mass/atom | expected | over-emission |
|---|---|---|---|---|
| C | +0.65 | 48.7 | 48.8 | **1.0x** |
| O | +0.48 | 84.6 | 34.9 | 2.4x |
| N | +0.45 | 74.8 | 37.0 | 2.0x |
| F | +0.40 | 124 | 31.5 | 3.9x |
| **S** | **+0.02** | 218 | 57.9 | 3.8x |
| **Cl** | **-0.09** | 205 | 53.2 | 3.9x |
| **I / Br** | +0.01 / +0.05 | — | — | — |

Carbon is essentially perfectly calibrated (48.7 mass/atom vs 48.8 from a 1.7 A vdW sphere at
0.75 A voxels). Everything else is over-emitted 2-4x, and **S/Cl/Br/I carry no signal at all** —
Poc2Mol emits a constant diffuse smear there regardless of the ligand. This is the mechanism
§3c predicted: an all-zero target channel gives `dice_coef == 0` with **zero gradient**, so the
channels that are usually empty were never taught to be empty.

### 12c. The likelihood/size correlation FLIPPED SIGN vs the published model

§5.1 measured `corr(ligand mean likelihood, heavy-atom count) = -0.758` on the published decoder.
On this decoder and panel it is **+0.403**, and **+0.604** against token count. Jude predicted
this: a longer SMILES gives the autoregressive decoder more context, so late tokens get cheaper
and the per-token mean RISES with length. **Do not assume the sign — measure it per model.**

The z-normalisation rationale is unaffected, because it rests on the size of the per-ligand
effect, not its direction. Variance decomposition replicates almost exactly:
pocket 8.2% / **ligand 83.9%** / interaction 7.9%, against §5.1's 6.4 / 84.0 / 9.5.

### 12d. Decoder and composition are nearly ORTHOGONAL

Mean within-pocket `corr(decoder score, composition score)` = **+0.024**. Residualising either on
the other barely dents it: decoder 0.7225 -> 0.6816, composition 0.7615 -> 0.7473. They reach
similar AUC through almost independent information, which is why the ensemble (0.785) beats both.
So "the decoder adds nothing over a formula readout" is **wrong** — it adds something different.

### 12e. Protein channels contribute ~nothing to ranking — masked ablation

Zeroing channels 9: (the 4 protein channels AND the presence flag, i.e. exactly
`assemble_decoder_input`'s keep=False branch):

| | shown | masked | delta |
|---|---|---|---|
| decoder z-norm | 0.7225 | 0.7171 | **-0.005** |
| decoder raw | 0.5467 | 0.5486 | +0.002 |
| residual vs composition | 0.6816 | 0.6847 | +0.003 |

**The decoder's ranking signal survives the pocket being removed.** Two consequences:

1. Its orthogonal-to-composition signal comes from the **predicted ligand density**, not from the
   pocket — which is the first positive evidence that the density carries usable structural
   information beyond size and composition.
2. The experiment-1 intervention is, at this checkpoint, contributing nothing to likelihood
   ranking.

⚠️ Caveats on 2: step 4000 is early and the predicted-density ramp was only ~15% complete;
`protein_mask_probability: 0.25` explicitly trains the model not to *depend* on the pocket; and
this is a **ranking** metric — the channels could still matter for generation (exact match,
Tanimoto). Re-run on a converged checkpoint before treating it as the experiment-1 result.

### 12f. Next test for the spatial question

Whether the density localises atoms is still unmeasured. The cheap decisive control: voxelise the
true ligand in its true pose and in a randomly rotated copy, then compare Dice against the
prediction. Same molecule, so size and composition are identical by construction and any gap is
pure spatial information — no docking and no decoy poses needed. CrossDocked (true binder pose vs
cross-docked non-binder in the same frame) is the full version, but pose quality is a confound
there and it overlaps PDBbind/HiQBind, so it needs a contamination check.

### 12g. RESULT — the density IS pose-specific (rotation control, 2026-08-10)

`scripts/adhoc_analysis/poc2mol_rotation_control.py`, 104 PLINDER-panel pockets. Compares the
prediction against the true ligand in its true pose and against a ROTATED COPY OF THE SAME
MOLECULE, so composition and size are identical by construction and any gap is purely spatial.
Rotation is about the box centre, which preserves distance from the centre, so no atom leaves the
24 A box — no truncation artifact.

| rotation | dice(pred, rot) | dice(true, rot) | win rate |
|---|---|---|---|
| true pose | **0.5964** | 1.000 | — |
| 15 deg | 0.5558 | 0.776 | **0.856** |
| 30 deg | 0.4622 | 0.552 | **0.990** |
| 45 deg | 0.3920 | 0.414 | 1.000 |
| 90 deg | 0.2627 | 0.227 | 1.000 |
| random | 0.2531 | 0.216 | 1.000 |

**The true pose beats a 15 deg rotation in 85.6% of pockets and a 30 deg rotation in 99%.** So
Poc2Mol localises atoms; it is not merely reporting a formula. This is the direct evidence behind
§12d/§12e — the decoder's ranking signal is orthogonal to composition and survives the pocket
being masked because there is real spatial structure in the density for it to read.

**Precision, in interpretable units:** `dice(pred, true) = 0.5964` falls between
`dice(true, rot@15)` = 0.776 and `dice(true, rot@30)` = 0.552, so the prediction is about as close
to the truth as a **~27 degree rigid rotation** of the true ligand. Real signal, but coarse.

⚠️ At 60 deg and beyond, `dice(pred, rot)` EXCEEDS `dice(true, rot)` (0.347 vs 0.334; 0.263 vs
0.227 at 90). The prediction overlaps a wrong pose better than the truth does — the signature of a
blurred, over-diffuse field, corroborating §12b's 2-4x over-emission.

### Where 12a-12g leave the diagnosis

1. The density carries genuine spatial information (12g) — the decoder's non-composition signal
   has something real to come from.
2. The density is smeared and mis-calibrated outside carbon (12b).
3. The decoder converts all of it into a ranking no better than a 9-number formula match (12a)
   and reconstructs almost nothing (0.5% exact match).

So the spatial signal exists and is being used poorly. **Sharpening / recalibrating Poc2Mol's
output is the highest-value next move** — training-free to test, targets something now measured to
be real, and the ~27 deg figure is the yardstick to measure any fix against. Decoder
regularisation targets a ceiling that 12a shows is not currently binding.

## 13. Experiment 1 A/B — protein channels, matched init (launched 2026-08-10)

W&B group `exp1_ab_20260810_115617`, nebius2. Both stage-3 runs from §11 killed at ~20k/12k steps
once §12 showed they were overfitting well past their optimum.

| arm | channels | init | GPUs | W&B |
|---|---|---|---|---|
| `protein` | 14 | `s1_maxagg_last.ckpt` | 0,1 | [ugc2md0w](https://wandb.ai/cath/voxelSmiles/runs/ugc2md0w) |
| `baseline` | 9 | `s1_maxagg_last_9ch.ckpt` | 2,3 | [v1q25zqv](https://wandb.ai/cath/voxelSmiles/runs/v1q25zqv) |

### The matched-initialisation problem, and the fix

The only genuine 9-channel stage-1 checkpoint is from the pre-2026-08-07 era that plateaued at
`val/loss` 0.19, against the 14-channel arm's 0.019 (§10d). Starting the arms from those would
have measured **stage-1 quality, not protein channels**.

`scripts/derive_9ch_checkpoint.py` instead slices the patch-embedding conv
(`model.encoder.embeddings.patch_embeddings.projection.weight`, (768, 14, 4, 4, 4)) to its first
9 input channels. A Conv3d is linear in input channels, so a 14-channel model fed zeroed protein
is *exactly* a 9-channel model with `W[:, :9]`. Verified rather than assumed: max |logit
difference| between the two on identical ligand input is **0.000e+00**. Both arms therefore start
from the same function, and the only difference is whether the pocket is visible.

### Schedule sized from the data, not inherited

`val/poc2mol/loss` bottomed at ~768k complex samples seen. At effective batch 300 with
`prob_poc2mol 0.5` that is 150 complex samples/step, i.e. **~5,100 steps** — so:

* `max_steps` 8000 (was 60000), decay over 4000-8000 so the anneal lands on the expected optimum
* predicted-density ramp **200 -> 2000** (was 2000 -> 15000), so the model reaches the deployed
  condition early instead of at the point it has already started overfitting
* `val_check_interval` **250** (was 2000) so the optimum is not missed, with
  `limit_val_batches: 10` to keep that affordable — the ZINC val set is 104,029 molecules and a
  full traversal every 250 steps would dominate wall-clock
* batch **150/GPU**, 2 GPUs per arm -> effective 300 (⚠️ NOT 600: 4 GPUs total across both arms,
  so each arm gets 2. Use 4 each if all 8 GPUs are free.)

### Checkpoint monitor changed: `val/loss` -> `val/poc2mol/loss`

`val/loss` is the pooled `combined` figure and ~99% ZINC *by sample count* (104,029 vs ~1,123
pocket samples). ZINC keeps improving while the pocket task degrades, so the two can diverge; they
happened to share a minimum in the §11 run, which was luck rather than design. Set in both s3
configs along with `optimized_metric` and a `pocloss` filename template.

⚠️ Hydra rejects `{` in an override value, so the filename template must live in the YAML — it
cannot be passed on the command line.

## 14. Experiment 1 A/B RESULT + hyperparameter sweep (2026-08-10)

### 14a. Protein channels do NOT help — decision: drop them

Both arms from the same stage-1 model (§13), 8k steps, matched everything:

| arm | best `val/poc2mol/loss` | best `likelihood_auc_znorm` |
|---|---|---|
| protein (14ch) | 0.5379 @999 | 0.7427 @1499 |
| **baseline (9ch)** | **0.5311 @1499** | **0.7759 @1249** |

The pocket channels make it *worse* on both. Expected given §13: their conv weights got zero
gradient during stage 1 (verified — channels 9-13 were exactly 0.0 and the flag exactly 0.0 for
every ZINC sample), so they enter stage 3 at random init and have ~1300 useful steps to learn
while 25% of their exposure is masked. **This is a fair test of "do protein channels help at this
fine-tuning stage", not of "would they help if pretrained properly".**

⚠️ For context: the parameter-free composition readout scores **0.7615** (§12a). The decoder's
0.7759 is only **+0.015** over reading the molecular formula off the density.

### 14b. Overfitting arrives at ~1300 steps

`val/poc2mol/loss` bottoms at step ~1249-1499, i.e. ~37k Poc2Mol samples seen — far earlier than
the 5,100 estimated in §13. Everything after is overfitting on 9,872 clusters.

### 14c. Two knobs that were never actually available

* **`weight_decay` was never settable.** `configure_optimizers` called
  `AdamW(self.parameters(), lr=...)` with no weight_decay, so EVERY run in this project has used
  the 0.01 default by accident. Now a config field and swept. Verified 0.01/0.1 reach AdamW.
* **Early stopping** on `val/poc2mol/loss` (patience 6 validation checks), in
  `configs/experiment/exp1_s3_baseline.yaml`. Verified firing: a patience-1 smoke run with
  `max_steps=200` stopped at step 40. This removes `max_steps` as a confound between sweep configs.

⚠️ Latent quirk found while testing: `scheduler_config.get("num_training_steps",
self.trainer.estimated_stepping_batches)` evaluates its default eagerly, so it always touches
`self.trainer` and raises outside a Trainer. Harmless in training; it just blocks calling
`configure_optimizers` standalone.

### 14d. Sweep — group `sweep_s3_20260810_132309`

`scripts/sweep_s3.py`, 24 configs, 4 concurrent (2 GPUs each), early-stopped. Target metric
`val/likelihood_auc_znorm`; **bar to clear is 0.7759**, and anything under 0.7615 is worse than
not having a decoder for this purpose.

| axis | values | why |
|---|---|---|
| `lr` | 2e-5, 5e-5, 1e-4 | interacts with how fast overfitting arrives |
| `weight_decay` | 0.01, 0.1 | never deliberately set before today |
| ViT dropout | 0.0, 0.1 | encoder runs at 0.0 (the GPT-2 decoder already defaults to 0.1) |
| `prob_poc2mol` | 0.25, 0.5 | lower = each cluster revisited less often, more ZINC anchoring. The released model used 0.3 (§4b) |

Report with `python scripts/sweep_s3.py --report --group sweep_s3_20260810_132309`.

## 15. Raw HiQBind re-acquired — and the parquet coordinates are bfloat16-quantised (2026-08-10)

### 15a. Re-download, on nebius2

Source is documented in `scripts/create_hiqbind_dataset.py`: figshare article **27430305**
("HiQBind Dataset", DOI 10.6084/m9.figshare.27430305), downloaded 2025-04-21.
Licence **CC BY 4.0** — so derived data may be redistributed with attribution, and the release
can ship the parquet AND the build script rather than forcing a rebuild.

`hiqbind.tar.gz` (8.07 GB) has md5 **b6668cc3a54a5bd686683924037a18d6** in BOTH v2 (2025-02-15)
and v3 (2025-02-22), and both predate the original download — so what we now have is provably the
same tarball that produced the current parquet, and regeneration cannot silently change which
systems exist. Extracted to `~/plixer_outer/hiqbind/raw_data_hiq_sm` (36 GB, 17,725 PDB entries)
plus `raw_data_hiq_poly` (651 MB) and the metadata CSVs.

### 15b. SDF atom order MATCHES the parquet — the mapping problem dissolves

For 192 sampled train rows: SDF present **100%**, and `*_ligand_refined.sdf` atom order matches
`ligand_element_symbols` **100%**. So every per-atom property (aromaticity, H count, acceptor,
formal charge) can be read straight off the SDF and indexed against the stored coordinates. No
graph isomorphism, no bond perception, no approximation.

For contrast, the routes available WITHOUT the raw data: canonical-SMILES order matches the
parquet only **3.1%** of the time, and bond-order-agnostic substructure matching recovered only
43%. Re-downloading was the right call.

### 15c. 🚨 NEW BUG — stored coordinates are bfloat16-quantised

`ligand_coords` and `protein_coords` are stored as float16 arrays whose VALUES are bfloat16-
rounded. Verified: `max |parquet - bfloat16(sdf)| = 0.000000` exactly, while
`max |parquet - sdf(float64)| = 0.249`.

| | value |
|---|---|
| ligand coords, mean worst-case error per ligand | **0.141 A** |
| ligand coords, worst observed | **0.492 A** |
| protein coords, max magnitude | 133 A -> bfloat16 spacing **0.5 A**, max error **0.25 A** |
| distinct protein coordinate values | 2,299 of 77,841 |
| voxel size | 0.75 A |

**So the grid is built from coordinates carrying up to ~2/3 of a voxel of quantisation error, on
both the model's input (protein) and its target (ligand).**

This is NOT the same as §3b bug 3. That one fixed bfloat16 *arithmetic* in the voxeliser by
centring in float32 — a real fix, but it cannot recover precision that was already destroyed when
the parquet was written. The residual 0.66% voxel disagreement §3b measured after the fix was
against these already-quantised inputs.

Likely downstream consequence: it puts a floor under how sharply any target density can localise
atoms, which bears directly on §12g's finding that Poc2Mol's prediction is only as precise as a
~27 degree rotation and is visibly blurrier than the truth. Regenerating in float32 removes it.

### 15d. ZINC needs none of this

`../zinc20_parquet` stores a full `mol_block` (RDKit 3D molfile with bonds), so the graph, atom
order and coordinates are all present and consistent. Only HiQBind was written lossily.

### 15e. Regeneration plan

Store per-atom **features**, not channel ids, so the channel scheme stays a runtime config choice:
`element, is_aromatic, n_hydrogens, is_acceptor, formal_charge` + **float32** coords.

Safeguards, both non-optional:
1. Pin to the existing `system_id` list and carry `split` / `cluster` / `*_cluster_id` across
   unchanged, so every prior result stays comparable.
2. Validate the regenerated coords against the current parquet **after** bfloat16-rounding them —
   they must match exactly, which proves same-source-version end to end.

Agreed channel scheme (11): `C_ali, C_aro, N_H, N_noH, O_H, O_noH, S, Halogen(Cl/Br/I), F,
other(incl. P), HBA`. Dropped from the first draft: the HBD overlay (measured ~100% redundant with
N_H + O_H: 2.36 vs 2.35 per ligand) and a standalone P channel (folded into `other`). Kept HBA
because it is genuinely non-redundant — the acceptor definition excludes amide N (~0.5/ligand,
29% of N_noH) which the H-count split cannot.

### 15f. Regeneration DONE — `parquet_v2` (2026-08-10)

`scripts/regenerate_hiqbind_parquet.py` and `scripts/regenerate_zinc_parquet.py`. Originals left
untouched at `../hiqbind/parquet` and `../zinc20_parquet` so the two can be A/B'd.

New schema, shared by both datasets:
`ligand_coords` (**float32**, (3,N) xyz-major), `ligand_coords_shape`, `ligand_element_symbols`,
`ligand_is_aromatic`, `ligand_n_hydrogens`, `ligand_is_acceptor`, `ligand_formal_charge`
(+ HiQBind's protein columns and the pinned `system_id/smiles/split/cluster/*_cluster_id`).

Per-atom FEATURES, not channel ids — the 11-channel grouping stays a runtime config choice, so
changing it never requires reprocessing.

**ZINC** — 5,537 files, **8,963,333 molecules, 0 unparseable**, in **under 90 s** at 48 workers.
Split preservation verified exactly: `index_train.csv` 5,477 files / 8,859,304 molecules and
`index_val.csv` 60 files / 104,029 molecules, both **identical** to the originals, file sets
identical, train/val overlap 0. Coordinates and element symbols round-trip against the source
mol_blocks with **0 mismatches**. Size 5.9 GB -> 2.8 GB, because dropping `mol_block` more than
pays for float32 coords.

⚠️ `mol_block` is NOT carried into v2 — that is the point (it removes `Chem.MolFromMolBlock` from
the per-sample hot path, which §3h identifies as the stage-1 bottleneck at ~790 samples/s). The
original `../zinc20_parquet` is therefore the ARCHIVAL copy and must not be deleted: raw ZINC20
mol2 is gone (§3), so those mol_blocks are the only remaining source.

**HiQBind** — **31,431 / 31,431 rows written; missing 0, feat_fail 0, validation_fail 0**, in
~25 min at 48 workers. File counts match the originals exactly (793 train / 106 val / 84 test).
`feat_fail 0` means every ligand sanitized, so aromaticity and acceptor status are real for all of
them rather than silently defaulting. `validation_fail 0` means every regenerated coordinate, once
rounded back to bfloat16, reproduces the stored value EXACTLY — same source version confirmed end
to end, for ligand and protein alike. Size 955 MB -> 2.0 GB (float32 coords + 4 feature arrays).

**Two performance traps found while writing these**, both worth remembering:
* `ProcessPoolExecutor` created inside the per-file loop spawned 16 workers x 983 files, each
  re-importing torch + rdkit + docktgrid. Startup dwarfed the work (0.04 s/system), turning a
  ~20 min job into a projected ~10 days. Hoist the pool.
* Parallelise ZINC per FILE, not per molecule — one RDKit import amortised over ~1,600 molecules.

## 16. 11-channel scheme — implemented and validated at the core (2026-08-10)

### Done and tested

* **`UnifiedView.match` understands feature-derived tokens**: `C_aromatic`, `C_aliphatic`,
  `N_withH`, `N_noH`, `O_withH`, `O_noH`, `HBA` (`_FEATURE_TOKENS` in `voxelizer.py`). They read
  `molecular_complex.atom_features`; a complex without features raises a clear error naming
  parquet_v2 rather than silently mis-assigning atoms.
* **`LIGAND_CHANNELS_V2` / `LIGAND_CHANNEL_NAMES_V2`** in `voxelization/config.py`.
  ⚠️ The catch-all (`other`) MUST stay last — `ligand_last_channel_is_catch_all` inverts
  `ligand_chs[-1]`, so moving it would invert the wrong channel.
* **`attach_atom_features`** (`poc2mol/datasets.py`) offsets ligand features by the protein atom
  count, because MolecularComplex concatenates protein-then-ligand and the masks span both.
* **End-to-end verified** on `parquet_v2`, 24 complexes: grid (24, **15**, 32,32,32) = 4 protein
  + 11 ligand, every channel populated.

| channel | mean occ | % samples occupied |
|---|---|---|
| carbon_aliphatic | 0.01151 | 95.8% |
| carbon_aromatic | 0.01080 | 83.3% |
| nitrogen_with_h | 0.00212 | 62.5% |
| nitrogen_no_h | 0.00289 | 83.3% |
| oxygen_with_h | 0.00066 | 37.5% |
| oxygen_no_h | 0.00318 | 87.5% |
| sulfur | 0.00098 | 45.8% |
| halogen | 0.00017 | 8.3% |
| fluorine | 0.00071 | 16.7% |
| hbond_acceptor | 0.00674 | 95.8% |
| other | 0.00041 | 4.2% |

### NOT yet done — required before any of the three runs can launch

1. ~~**`ParquetVox2SmilesDataset` cannot read zinc v2.**~~ **Done 2026-08-10 — see §17.**
2. **Data configs** pointing at `parquet_v2` / `zinc20_parquet_v2` with the v2 channel map.
3. **Channel counts**: Poc2Mol `out_channels` 9 -> **11**; Vox2Smiles `num_channels` 9 -> **11**
   (baseline) or 11 + 4 + 1 = **16** (protein arm).
4. **Stage-1 weights do not transfer** — the patch-embedding conv changes width, so ZINC
   pretraining must be redone. Use the warm-start trick (the inverse of
   `derive_9ch_checkpoint.py`, which was verified bit-exact): copy old C -> {C_ali, C_aro},
   O -> {O_H, O_noH}, N -> {N_H, N_noH}, S/Halogen/F direct, and small-random for HBA/other.
5. **Launch scripts** for poc2mol / vox2smiles / combined on the v2 configs.

⚠️ Also unresolved from §14d: the sweep showed decoder hyperparameters do not move
`likelihood_auc_znorm` (all 24 configs within noise of the 0.7615 composition readout), so the
v2 data — float32 coords and the richer channels — is the actual hypothesis under test now.
Retraining Poc2Mol on v2 with the OLD 9 channels first would isolate what the coordinate
precision fix alone buys, measured against the 27-degree rotation yardstick from §12g.

## 17. Vox2Smiles on parquet_v2 — RDKit-free ligand path + stage-1 v2 runs (2026-08-10)

### The dataset change

`ParquetVox2SmilesDataset` could only build a sample by calling `Chem.MolFromMolBlock`, and v2
deliberately drops `mol_block` (§15f). Two new pieces:

* **`StoredLigandComplex`** (`voxelization/voxelizer.py`) — a ligand-only `MolecularComplex`
  built straight from stored arrays. Mirrors `RDkitMolecularComplex` with three deliberate
  differences: coords/radii/centre stay **float32** (`RDkitMolecularComplex` casts to
  docktgrid's `DTYPE`, which our patch sets to bfloat16 — i.e. it would re-impose exactly the
  quantisation §15c regenerated the data to remove), `atom_features` are carried so the
  11-channel scheme resolves, and no RDKit object exists at all.
* **`stored_molecule_atom_record`** (`vox2smiles/datasets.py`) — same augmentation order as
  `prepare_rdkit_molecule`, so v2 records are drop-in interchangeable with v1 ones.

`__getitem__` branches on **the absence of `mol_block`**, which is the honest discriminator
between the two layouts. Hydrogens are left in place rather than honouring
`include_hydrogens`: they belong to no channel (the catch-all enumerates H and is inverted),
so `atom_record_from_complex`'s `keep` mask drops them anyway.

### Verified, not assumed

| check | result |
|---|---|
| stored `smiles` == v1 `MolToSmiles(MolFromMolBlock(...))` | **300/300 byte-identical** |
| v1 vs v2 grid, 9ch, augmentation off | max 0.121, mean 3.1e-05, **0.0030%** of voxels >0.05 apart |
| **same, after rounding v2 coords back to bfloat16** | max 0.055, **0.0001%** >0.05 apart |
| atoms kept, v1 vs v2 | identical |

The third row is the control that makes the second interpretable: rounding v2's float32 coords
to bfloat16 collapses the difference by 30×, so the v1/v2 gap **is** the coordinate precision
the regeneration removed, not a structural discrepancy. (The residual is `ligand_center` and
`vdw_radii`, which v1 also computes in bfloat16.) The training target is unchanged — same
SMILES string, so v1 and v2 stage-1 runs are directly comparable.

### ⚠️ `other` is nearly dead on ZINC — 0.086%

Scanned 88,114 molecules across 40 v2 files: only **76 carry a non-{C,H,O,N,S,Cl,F,I,Br} atom**
(66 P, 10 Si). At batch 128 that is a populated `other` channel in ~10% of batches, so its
patch-embedding weights get little gradient during stage 1 — **the same setup that left the
protein channels near init and lost the §14a A/B**. Much milder (1 channel of 11, and `other`
reaches 4.2% on HiQBind per §16) but it is the channel to watch when stage 3 starts.

### Configs

| | 9-channel (control) | 11-channel |
|---|---|---|
| data | `vox2smiles_zinc_v2_9ch` | `vox2smiles_zinc_v2_11ch` |
| experiment | `exp1_zinc_v2_9ch` | `exp1_zinc_v2_11ch` |

The 11ch data config's ligand map is **byte-identical to `poc2mol_hiqbind_v2_11ch.yaml`** by
construction — they must agree or the decoder is pretrained on a different channel semantics
from what Poc2Mol emits, which is §3b bug 2 all over again.

**No protein channels** — §14a measured them hurting both metrics and the decision was to drop
them.

### Stage-1 v2 runs (launched 2026-08-10, nebius1)

`scripts/run_exp1_stage1_v2.sh`, group `exp1_zinc_v2_20260810_224156`,
logs `logs/exp1_stage1_v2/20260810_224156/`.

| arm | channels | GPUs | it/s | samples/s | ETA | W&B |
|---|---|---|---|---|---|---|
| `v2_9ch` | 9 | 0-3 | 1.84 | ~942 | ~75 h | [wbxm3u0d](https://wandb.ai/cath/voxelSmiles/runs/wbxm3u0d) |
| `v2_11ch` | 11 | 4-7 | 1.69 | ~865 | ~82 h | [d9ngjry8](https://wandb.ai/cath/voxelSmiles/runs/d9ngjry8) |

Schedule copied verbatim from §9c (lr 3e-4, WSD 2k/298k/200k, `min_lr_ratio` 0.03, `max_steps`
500k, batch 128/GPU × 4 = effective 512, `val_check_interval` 5000), so the **data** is the only
change against a known-good recipe. One ZINC epoch = 17,304 steps; the run is ~29 epochs.

**Both arms are run because a stage-1 checkpoint cannot cross channel schemes** — the encoder's
patch-embedding conv width is set by the channel count. Running only one would mean an ~80 h
wait after the Poc2Mol v2 A/B (nebius2, `q2txptiw` 9ch vs `2ho5efkl` 11ch) reports. The 9ch arm
also isolates what the **coordinate precision fix alone** buys, against §9c's v1 numbers.

Throughput is **up ~15%** on v1's 1.6 it/s despite the extra channels, because v2 removes
`MolFromMolBlock` from the hot path — the stage-1 bottleneck identified in §3h.

### Launch trap: `trainer/ddp.yaml` hardcodes `devices: 8`

Masking an arm to 4 GPUs with `CUDA_VISIBLE_DEVICES` is not enough — Lightning raises
`You requested gpu: [0..7] But your machine only has: [0..3]` rather than clamping. Every
per-arm launch must pass `trainer.devices=N` explicitly. **The §9c health check caught this**:
the first launch died at startup and the 240 s liveness probe reported `DEAD` with the log tail,
instead of leaving 8 GPUs idle behind a "launched" message.

## 18. Poc2Mol v2 A/B result + per-channel emission audit (2026-08-10/11)

### 18a. The `val/loss` gap between the two schemes is FLOOR, not quality

Poc2Mol v2 A/B, nebius2: `ch9` [q2txptiw](https://wandb.ai/cath/poc2mol/runs/q2txptiw) vs
`ch11` [2ho5efkl](https://wandb.ai/cath/poc2mol/runs/2ho5efkl). Both plateaued
(best 0.8039 @ep390 and 0.7275 @ep426; §3g's ~300-350 useful budget replicates).

`ch11` looks 0.075 better on `val/loss` — and discriminates **identically**:

| | ch9 | ch11 |
|---|---|---|
| best `val/loss` | 0.8073 | **0.7334** |
| composition AUC z-norm | **0.7421** | 0.7414 |
| composition AUC raw | 0.6754 | 0.6900 |
| size-only AUC z-norm | 0.5596 | 0.5669 |
| pocket-blind control | 0.4964 | 0.4979 |

The 11-channel scheme splits C/N/O into pairs that are usually BOTH occupied and adds an
always-occupied HBA channel, so a larger fraction of its channels are non-empty, which
mechanically lowers the Dice floor (§3c). **Comparing two channel schemes on `val/loss` is
meaningless.** Full val, 1019 pockets / 999 unique ligands.

`scripts/adhoc_analysis/poc2mol_scheme_discrimination.py` — scheme-agnostic (reads the channel
map from the checkpoint's own `resolved_config.yaml`), reports a per-scheme composition AUC and
a scheme-independent size-only AUC. ⚠️ The composition readout is NOT comparable across
schemes: 11 channels give it 11 descriptors against 9, so a richer scheme can win on descriptor
count alone. Panel here is HiQBind val, NOT §12's 104-pocket PLINDER panel, so 0.742 is not
comparable to §12a's 0.7615.

### 18b. EVERY channel over-emits — and §12b understated it

`scripts/adhoc_analysis/poc2mol_channel_emission.py`. Reference is the TRUE VOXELISED LIGAND on
the same grid, not §12b's analytic single-sphere estimate — which ignores that overlapping atoms
do not deposit additively under max-aggregation.

ch9 @ep357, full val:

| channel | ratio | empty% | empty_mass | occ_mass | slope | intercept | on-target |
|---|---|---|---|---|---|---|---|
| carbon | 1.33 | 0.1% | 303 | 807 | 0.52 | 488 | **58%** |
| oxygen | 2.26 | 9.1% | 254 | 364 | 0.81 | 227 | 24% |
| nitrogen | 1.73 | 8.7% | 157 | 249 | 0.46 | 177 | 25% |
| sulfur | 3.52 | 71% | 80 | 95 | 0.19 | 79 | 4.8% |
| chlorine | 4.36 | 88% | 44 | 62 | 0.19 | 44 | 2.7% |
| fluorine | 7.88 | 89% | 59 | 75 | 0.23 | 59 | 1.7% |
| iodine | **10.4** | 99.8% | 1.9 | 2.4 | 0.01 | 1.9 | 0.0% |
| bromine | 5.63 | 98% | 12 | 15 | 0.03 | 12 | 0.4% |
| other | 1.92 | 88% | 13 | 123 | **0.90** | 14 | 27% |

🚨 **§12b's "carbon is essentially perfectly calibrated (48.7 vs 48.8 mass/atom)" is WRONG.**
Against the true grid, carbon over-emits **1.33x** and only **58%** of its predicted mass lands
on real ligand density. The analytic reference flattered it.

**The mechanism is directly visible as a constant smear.** For every rare channel
`intercept ~ empty_mass ~ occ_mass`: fluorine emits **59.1** units when the ligand has NO
fluorine and **75.1** when it does, slope 0.23. Chlorine 43.5 vs 61.9; sulfur 79.7 vs 94.6.
Carbon shows it at scale too — intercept 488 against occupied mass 807, so **~60% of predicted
carbon mass is content-independent baseline**. This is §3c's zero-gradient hole made concrete:
an all-zero target channel gives Dice a numerator identically zero, so only BCE — at ~0.011 of
the loss against Dice's ~0.62 — charges the model for that mass.

### 18c. ⚠️ But sparsity does NOT cause the smear — `other` is the counterexample

`other` is empty in 88.3% of ligands, the SAME sparsity as chlorine, and behaves completely
differently:

| | chlorine | other |
|---|---|---|
| empty% | 88.2% | 88.3% |
| empty_mass / occ_mass | 43.5 / 61.9 = **0.70** | 12.8 / 123.0 = **0.10** |
| slope | 0.19 | **0.90** |
| r | 0.266 | **0.712** |

`other` is the best-calibrated channel in the model. So a sparse channel CAN be learned, and the
zero-gradient hole **permits** the smear without forcing it. Where the pocket genuinely
determines the answer (phosphate-binding pockets are highly distinctive) the model commits;
where it does not (halogen placement) it hedges with a constant, and nothing charges it.

**Do not attribute the rare-channel failure purely to the loss.** Part of it is that halogen
position is genuinely hard to predict from the pocket.

### 18d. 11ch: halogen merge helps, carbon split hurts

| | ch9 | ch11 |
|---|---|---|
| halogens | Cl 4.36 / F 7.88 / I 10.4 / Br 5.63 | **halogen 3.96** / F 5.88 |
| halogen r | 0.027-0.27 | **0.294** |
| carbon | 1.33 | C_ali **1.78** / C_aro 1.45 |

Merging Cl/Br/I did what §16 designed it for. But splitting carbon made BOTH halves worse than
the single channel — aliphatic/aromatic assignment is harder to predict than carbon position,
and each half carries its own smear. Consistent with the two schemes tying on discrimination.

### 18e. `val/emission/*` now logged during validation

`ratio`, `on_target`, `empty_frac` in `Poc2Mol.validation_step` — three reductions, no extra
forward. These are NOT floor-bound, unlike `val/loss`, and `empty_frac` is the direct target of
any fix. ⚠️ **Read `on_target` alongside `empty_frac`**: a model that simply emits less
everywhere lowers `empty_frac` without getting sharper. Applies to NEW runs only.

### 18f. LR schedule — warmup is 83 EPOCHS, and there is no stable phase

HiQBind is 9,872 clusters, so at effective batch 1536 an epoch is only **~6 optimiser steps**.
The v2 runs' `num_warmup_steps: 500` is therefore **83 epochs** of a 600-epoch run (14%), after
which `cosine_with_min_lr` decays across the *entire* remainder to `min_lr_rate: 0.5`. There is
no stable phase at any point. Always convert step-denominated schedules into epochs for this
dataset before trusting them — 9,872 clusters makes steps/epoch tiny and step counts misleading.

### 18g. BCE-weight sweep (launched 2026-08-11, nebius2)

`scripts/run_poc2mol_bce_sweep.sh`, group `poc2mol_bce_sweep_<stamp>`. 4 arms x 2 GPUs, ch11
scheme, effective batch 1536 (matched to the v2 runs via accumulation 2), `max_epochs` 450.

| arm | change |
|---|---|
| `a1_ctrl` | alpha 1.0 — control |
| `a10` | alpha 10 |
| `a30` | alpha 30 |
| `sched` | alpha 1.0, warmup 500 -> **50** steps, `min_lr_rate` 0.5 -> **0.1** |

alpha 10/30 chosen because BCE at alpha 1 is ~56x smaller than Dice; 10-30 brings it to the same
order. **Judge on `val/emission/{empty_frac,on_target}`, not `val/loss`** — the floor dominates
it AND shifts with alpha, so cross-arm `val/loss` is doubly meaningless here.

### 18h. 🚨 `val/loss` was ANTI-CORRELATED with discrimination — ch11, epochs 366 -> 576

Both v2 runs completed 600 epochs. ch9 best 0.8039 @ep390 (final 0.8113, drift +0.0074);
ch11 best **0.7265 @ep576** (final 0.7292) — ch11 was still improving at the end while ch9
turned at 390.

Re-running the discrimination test on the BEST checkpoints reverses the §18a picture:

| | ch9 | ch11 |
|---|---|---|
| composition AUC z-norm @mid-run | 0.7421 (ep357) | 0.7414 (ep366) |
| composition AUC z-norm @**best** | **0.7515** (ep390) | **0.7295** (ep576) |
| `val/loss` @best | 0.8039 | 0.7265 |

**ch11 improved `val/loss` by 0.0069 over epochs 366->576 while LOSING 0.0119 of
discrimination.** This is a WITHIN-run, within-scheme comparison — same channel map, same
floor, two checkpoints of one job — so unlike §18a there is no floor confound whatsoever.

**Consequence: do NOT select Poc2Mol checkpoints on `val/loss`.** It is not merely
insensitive (§3c); here it actively picked a worse model. `ModelCheckpoint` monitoring
`val/loss` did exactly that for ch11. Either monitor a downstream metric or fix the epoch
budget from §3g (~300-350) rather than trusting the curve.

⚠️ The 0.7515 vs 0.7295 gap between schemes is ~2 SE on 1019 pockets and unpaired — suggestive,
not settled. The within-run decoupling is the solid part. A paired per-pocket test would
settle the between-scheme question; `poc2mol_scheme_discrimination.py` currently returns only
aggregates.

### 18i. BCE-weight sweep 1 RESULT — alpha helps, the LR schedule does not (2026-08-11)

Group `poc2mol_bce_sweep_20260811_000434`, 4 arms x 2 GPUs, ch11 scheme, 450 epochs, effective
batch 1536. Ranked on discrimination + emission, NOT `val/loss` (§18h).

| arm | disc z-norm | C_ali ratio | C_aro ratio | halogen | fluorine | `val/loss` |
|---|---|---|---|---|---|---|
| `a1_ctrl` | 0.7364 | 1.88 | 1.51 | 4.07 | 6.47 | 0.7340 |
| `a10` | 0.7384 | 1.41 | 1.28 | 3.13 | 4.26 | 0.8411 |
| **`a30`** | **0.7504** | **1.36** | **1.10** | **2.34** | **3.59** | 1.0579 |
| `sched` | 0.7361 | 1.89 | 1.55 | 4.21 | 7.77 | 0.7405 |

**Over-emission falls monotonically in alpha on EVERY channel, and `on_target` RISES at the same
time** — C_aro 45.5% -> 48.0% -> 49.9%, halogen 3.9% -> 4.5% -> 5.5%, sulfur 5.2% -> 5.7% ->
6.6%. So this is not the degenerate "emit less everywhere" win that §18e warned about; the mass
that remains is better placed. The guard metric did its job.

⚠️ **But BCE does NOT fix selectivity.** `empty_mass / occ_mass` barely moves (C_ali 0.716 ->
0.716 -> 0.682; fluorine 0.80 -> 0.87 -> 0.79). The model emits proportionally less
*everywhere* rather than learning which channels the ligand leaves empty. BCE sharpens
calibration; it does not close the zero-gradient hole. A real fix needs the loss to charge for
empty-channel mass in a way Dice currently cannot.

**The LR schedule is NOT the problem.** `sched` (warmup 500->50 steps, `min_lr_rate` 0.5->0.1)
scored 0.7361 against the control's 0.7364 -- identical -- with slightly WORSE emission, despite
an enormous early-epoch lead (at epoch 18 it had ratio 3.9 / on_target 0.157 while the arms
still in warmup had ratio 37 / on_target 0.005). The 83-epoch warmup costs nothing in final
quality; it only delays early progress. Consistent with §3d. **Leave the schedule alone.**

`val/loss` across arms is meaningless by construction here (alpha multiplies the BCE term), as
the table shows -- a30 is the best model and has the WORST `val/loss` by a wide margin.

⚠️ Significance: a30's +0.014 over control is only ~1.5-2 SE unpaired, and a30's 0.7504 merely
MATCHES the best 9-channel model (0.7515, §18h) rather than beating it. The monotone trend
across two independent metric families is the stronger evidence. Sweep 2 (group
`poc2mol_bce_sweep2_20260811_035257`) extends to alpha 100/300 and adds **seed replicates of
a30 and a1** -- without those, "alpha 30 helps" is not separable from seed noise.

### 18j. Sweep 2 — 🚨 the discrimination gain was SEED NOISE; the calibration fix is real

Group `poc2mol_bce_sweep2_20260811_035257`. Two arms are seed replicates of sweep 1's endpoints.

**Discrimination (composition AUC z-norm), all seven runs:**

| alpha | seed 42 | seed 1 | within-condition spread |
|---|---|---|---|
| 1 | 0.7364 | 0.7461 | **0.0097** |
| 10 | 0.7384 | — | |
| 30 | 0.7504 | 0.7449 | **0.0055** |
| 100 | 0.7413 | — | |
| 300 | 0.7540 | — | |

alpha-1 mean 0.7413, alpha-30 mean 0.7477 -> **+0.0064, smaller than the 0.0097 seed spread of
the control condition alone.** §18i's "+0.014 for a30" was seed noise. There is no reliable
discrimination trend in alpha (a100 scores BELOW a30). **Do not claim BCE weight improves
ranking.**

**Calibration, by contrast, is a huge, monotone, far-outside-noise effect:**

| channel ratio | a1 (42/1) | a10 | a30 (42/1) | a100 | a300 |
|---|---|---|---|---|---|
| carbon_aliphatic | 1.88 / 1.84 | 1.41 | 1.36 / 1.34 | 1.25 | **1.11** |
| carbon_aromatic | 1.51 / 1.47 | 1.28 | 1.10 / 1.17 | **1.00** | 1.08 |
| sulfur | 3.26 / 3.60 | 2.68 | 2.14 / 2.16 | 1.44 | **1.12** |
| halogen | 4.07 / 3.52 | 3.13 | 2.34 / 2.51 | **1.42** | 1.55 |
| fluorine | 6.47 / 4.18 | 4.26 | 3.59 / 3.30 | **1.80** | 1.92 |

The two alpha-1 seeds agree tightly (1.88/1.84 carbon), so this is signal, not noise. **At
alpha 100-300 the model is essentially calibrated** (ratios 1.0-1.9) against 1.5-6.5 at alpha 1.
`on_target` also rises ~3-4 points and selectivity improves for the COMMON channels
(`empty/occ` for C_ali 0.71 -> 0.65 -> 0.62), though not for the rare ones (fluorine stays ~0.8).

### 18k. Why the metric could never have rewarded this — and where the benefit actually is

`poc2mol_scheme_discrimination.py` calibrates predicted mass to atom-count units by fitting a
per-channel `scale` before scoring. **A constant per-channel over-emission factor is therefore
divided straight out**, so the composition readout is scale-invariant BY CONSTRUCTION and
cannot reward a calibration fix. The flat discrimination result is what that metric must
produce, not evidence the fix is worthless.

Where it should matter is the DECODER. Vox2Smiles stage 1 pretrains on TRUE ligand grids
(ratio 1.0 by definition); stage 3 feeds it Poc2Mol density. At alpha 1 that density carries
2-6x the mass the decoder was trained on -- a large distribution shift, of exactly the kind
§3c warns about for `max_poc2mol_loss`. alpha 100-300 removes it at no measured cost
(discrimination flat, not worse).

**Recommendation: train the production Poc2Mol at alpha ~100.** a100 has the best carbon
calibration (1.00 aromatic) and the best halogen/fluorine ratios; a300 is marginally better on
some channels but pushes `val/loss` to 3.79 and starts degrading `other` (slope 0.94 vs a30_s1's
0.72 -- worth a look). ⚠️ The claim to TEST, not assume, is that better-calibrated density
improves the decoder; nothing measured here demonstrates that, and it needs a stage-3 run.

### 18l. ⚠️ CORRECTION to 18k — alpha degrades Dice monotonically; use alpha 10, NOT 100

18k recommended alpha ~100 on calibration alone. That was premature: `val/dice` is logged
separately and IS comparable across arms (beta=1 in every arm, floor constant within scheme),
and it degrades monotonically with alpha.

| alpha | dice_min (seeds) | real error (-0.6403 floor) | vs alpha 1 | C_ali ratio |
|---|---|---|---|---|
| 1 | 0.7231 (0.7212/0.7249) | 0.0828 | — | 1.86 |
| 10 | 0.7241 | 0.0838 | **+1.2%** | 1.41 |
| 30 | 0.7295 (0.7280/0.7310) | 0.0892 | +7.7% | 1.35 |
| 100 | 0.7327 | 0.0924 | +11.6% | 1.25 |
| 300 | 0.7410 | 0.1007 | **+21.6%** | 1.11 |

Seed spreads are 0.0030-0.0037, so alpha 300's degradation is ~5x noise. **The trade is real:
calibration up, spatial overlap down, ranking flat.** Overlap is the part §12g established is
genuinely pose-specific, so paying 11-22% more overlap error for a calibration benefit that is
still HYPOTHETICAL (no stage-3 run has tested it) is a bad trade.

**Use alpha 10** — ~half the calibration gain (1.86 -> 1.41) for +1.2% Dice error, inside seed
noise. alpha 30 is the outer limit. Revisit only if a stage-3 run demonstrates that calibrated
density measurably helps the decoder.

Lesson: `val/loss` is unusable across alphas, but its `val/dice` COMPONENT is comparable and was
the piece that mattered. Log and read components, not just totals.

## 19. Stage-3 A/B, 9ch vs 11ch full pipeline (2026-08-11)

Matched pipelines: `poc2mol_v2_{9,11}ch` best-val/loss checkpoint + the `{9,11}`-channel stage-1
decoder (`epoch_003.ckpt`, matched steps), `inject_protein: false` (§14a), ramp 200->2000,
`prob_poc2mol` 0.5. Configs `exp1_s3_v2_{9,11}ch`, data `vox2smiles_combined_v2_{9,11}ch`.

### 19a. 🚨 Run 1 stopped on the WRONG metric — 11ch was killed mid-ascent

Group `exp1_s3_v2_20260811_075508`. Early stopping monitored `val/poc2mol/loss`, which is NOT
the objective:

| step | 9ch AUC znorm | 11ch AUC znorm |
|---|---|---|
| 1499 | **0.7342** (peak) | 0.7045 |
| 1999 | 0.6961 | 0.7305 |
| 2499 | 0.6650 | 0.7446 |
| 2999 | (stopped @2749: 0.6592) | **0.7457, still rising** |

`val/poc2mol/loss` bottomed at step 1499 (11ch) / 1249 (9ch) and rose thereafter, so early
stopping fired while `likelihood_auc_znorm` was still climbing. **This is CLAUDE.md §18h one
stage down** — the cross-entropy and the ranking metric decouple, and stopping on the loss
selects the wrong checkpoint. Should have been anticipated after §18h.

**The two schemes behave qualitatively differently**, which is the real finding: 9ch peaks at
~1500 then falls off sharply (0.7342 -> 0.6592, i.e. -0.075), while 11ch climbs monotonically.
11ch is ahead (0.7457 vs 0.7342) WITHOUT having converged.

`val/zinc/loss` was flat throughout for both (0.0128-0.0148), so no pretraining regression --
and 11ch sits consistently lower (0.0128 vs 0.0147), consistent with its stage-1 lead.

### 19b. Run 2 — corrected criterion

Group `exp1_s3_v2b_20260811_091246`. Early stopping AND checkpoint selection now monitor
`val/likelihood_auc_znorm` (mode max, patience 12 -- the metric swings +-0.02 between adjacent
checks on the 104-pocket panel, so patience must be generous). `max_steps` 4000 -> 12000,
schedule warmup 200 / stable 5800 / decay 6000.

⚠️ **Two confounds to state whenever this A/B is quoted:**
1. The 11ch stage-1 decoder is AHEAD on its own pretraining task at matched steps (zero-shot
   zinc exact match 0.656 vs 0.375). So an 11ch win may reflect its channels being easier to
   pretrain rather than better Poc2Mol density. This compares whole PIPELINES, not densities.
2. Checkpoint selection handicaps 11ch by ~0.012 (§18h): best-val/loss picks ch11 ep576, which
   discriminates 0.7295 against ep366's 0.7414.
3. `likelihood_auc_znorm` best-over-N-validations is subject to maximum-selection bias, which
   grows with the number of validation checks. Compare at matched check counts or use a
   smoothed value, not the raw max.

### 19c. RESULT — BCE calibration does NOT help the decoder (§18k answered negative)

Group `exp1_s3_bce_20260811_095929`. Four stage-3 arms sharing the SAME 11ch decoder init
(`s1_v2_11ch.ckpt`), same schedule, same data. Only variable: the BCE weight the frozen
Poc2Mol was trained with.

| arm | raw max | smoothed max | last-5 | poc2mol/tanimoto |
|---|---|---|---|---|
| `a1_ctrl` | 0.7353 | 0.7203 | 0.7026 | 0.1714 |
| `a10` | 0.7304 | **0.7053** | **0.6882** | 0.1661 |
| `a30` | **0.7599** | 0.7286 | 0.7039 | **0.1739** |
| `a100` | 0.7438 | **0.7307** | **0.7064** | 0.1679 |

**Non-monotonic and within noise.** `a10` is BELOW the control on every measure; the total
spread (0.025 smoothed) is the size of the +-0.02 check-to-check swing on the 104-pocket panel.
Taking over-emission from 1.88x to 1.25x buys nothing downstream.

**§18k's distribution-shift argument is therefore rejected**, and **§18l's "use alpha 10"
recommendation is not supported** -- a10 is the worst arm here. Keep alpha at 1 unless some
other downstream metric argues otherwise. The calibration effect itself is real (§18j) but has
no measured consumer.

⚠️ Arms had different validation counts (25-31), so raw max carries unequal selection bias.
a10 had the MOST checks and still scored lowest, so its poor showing is not a selection artifact.

### 19d. The schedule fix WAS worth it — tanimoto +15%

Landing the LR decay on the observed optimum (warmup 200 / stable 1800 / decay 2000, so decay
starts at step 2000) rather than at step 6000 lifted `val/poc2mol/tanimoto` from **0.147** in the
§19b A/B to **0.166-0.174** in all four arms here. Larger and more consistent than anything the
BCE weight moved. In §19b decay started at 6000 while early stopping fired at ~4500-6000, so no
arm ever saw the anneal. **Size the decay to the observed optimum, not to max_steps.**

## 20. Ensembling — model >> augmentation, subset selection buys nothing (2026-08-11)

`scripts/adhoc_analysis/poc2mol_ensembling.py` + `ensemble_subset_search.py`. 7 Poc2Mol 11ch
checkpoints x 4 augmentation replicates = 28 members, full 1019-pocket HiQBind val, composition
readout. Members are COLUMN z-normalised once before averaging (the same transform
`likelihood_auc_znorm` applies), so no member dominates via dynamic range and the single-member
baseline is computed on the same scale.

| | AUC | gain |
|---|---|---|
| single member | 0.7253 (sd 0.0092) | — |
| + augmentation (x4) | 0.7419 | +0.0166 |
| + model (x7) | 0.7555 | **+0.0302** |
| + both (7x4) | 0.7578 | +0.0325 |

**Strongly sub-additive.** Marginal gain of augmentation ON TOP of model ensembling is only
**+0.0023**; model on top of augmentation is +0.0159. The two average largely the same noise.
**Do model ensembling; augmentation only matters if you have one model.** Member-member score
correlation 0.715 (0.624-0.833).

Per model (aug-collapsed): a30_s1 0.7521, a1_s1 0.7498, a30 0.7478, a100 0.7426, a1_ctrl 0.7385,
sched 0.7359, **v2main 0.7268 (worst)**. v2main is the ch11 ep576 checkpoint — replicating
§18h's 0.7295, i.e. best-val/loss really does select a poor discriminator.

**Subset selection is noise-fitting**: greedy 0.7600 (selected and scored on the same pockets),
split-half honest estimate 0.7586, average-everything 0.7578. Spread 0.002 against a member sd
of 0.0092. **Average all members.** Greedy did pick the two SEED replicates, consistent with
seed diversity being what pays.

⚠️ 🚨 **The first version of this measured nothing on the augmentation axis.** The Poc2Mol val
set is deterministic by design (§3e: `rotate: false, translation: 0.0`), so the replicates were
bitwise identical — the tell was `model_ensemble == aug_x_model` to 16 digits. The script now
takes `--rotate` and prints `max |mass(aug0) - mass(aug1)|`, flagging it when zero.
A second bug in that version: members were ROW-standardised before averaging while the metric
z-normalises by COLUMN, which inflated every ensemble figure by ~+0.005 relative to its
baseline. Normalise members with the same transform the metric uses.

**Not included here: the mass x decoder fusion axis** (§12d, +0.024 to 0.785). Those two
readouts correlate at only r = +0.024 against 0.715 among these members, so fusion is the axis
most likely to be genuinely additive on top of this ensemble — the obvious next test.

## 21. 🏆 BEST DECODER CHECKPOINT ON DISK — and it nearly wasn't saved

```
nebius2:~/plixer_outer/plixer/checkpoints/best_decoder/
    s3_ab_baseline_step1250_auc0.7759.ckpt     <- the one
    s3_ab_baseline_step1500_auc0.7544.ckpt     <- runner-up / fallback
```

`val/likelihood_auc_znorm` = **0.7759** at global_step 1249, from run
[v1q25zqv](https://wandb.ai/cath/voxelSmiles/runs/v1q25zqv) (`ab_baseline`, group
`exp1_ab_20260810_115617`, §14a). Full optimiser state, 1.03 GB.

**Pipeline it belongs to — v1, NOT v2.** 9-channel decoder, v1 `../hiqbind/parquet` +
`../zinc20_parquet`, and the OLD shared Poc2Mol
`checkpoints/exp1_shared_poc2mol/poc2mol_cons4_drop_epoch307.ckpt`. It cannot be loaded against
the v2 configs — different channel semantics and different upstream. Reproduce with
`configs/experiment/exp1_s3_baseline.yaml`.

| all decoder checkpoints on disk | AUC | pipeline |
|---|---|---|
| **`ab_baseline` step 1250** | **0.7759** | v1, 9ch |
| `s3_bce a30` | 0.7599 | v2, 11ch |
| `exp1_s3_v2b` 11ch | 0.7576 | v2, 11ch |
| `ab_protein` step 1500 | 0.7427 | v1, 14ch |

⚠️ **Treat 0.7759 as an optimistic point estimate, not a converged level.** Adjacent validation
checks are 0.7150 (step 999) and 0.7544 (step 1499) -- swings of +-0.03 -- and it is the max
over ~27 checks, so it carries maximum-selection bias (§19b caveat 3). The honest level is
nearer 0.74-0.75. It is nonetheless the ONLY decoder we have that beats the parameter-free
composition readout (0.7615, §12a), and then by only +0.014.

🚨 **It survived by luck.** Checkpointing monitored `val/poc2mol/loss`, which happened to sit
near its own minimum at the same step. One validation check either side and the peak would have
been lost. This is §18h/§19a biting for real: **every run before 2026-08-11 checkpointed on a
loss that decouples from the ranking metric**, so any historical run whose AUC peaked away from
its loss minimum has already lost that checkpoint. The v2 stage-3 configs now monitor
`val/likelihood_auc_znorm` directly.

---

# 22. Corrections and baselines carried back from the generative branch (2026-08-12)

The `generative-poc2mol` branch was concluded as a negative result (its own §22 lives on
that branch). Four findings there are **not** generative-specific and correct or extend the
record above.

## 22a. §12g's headline pathology does NOT reproduce on the current checkpoint

§12g reports that past 60 deg the prediction overlaps a WRONG pose better than the true one
(0.347 vs 0.334, win rate below 0.5) — the signature of a blurred field, and the finding
that motivated a great deal of subsequent work. Re-measured on **HiQBind v2 val** with
`poc2mol_v2_11ch_ep576`, the same rotation control gives:

| rotation | dice(pred, rot) | win rate |
|---|---|---|
| true pose | 0.4976 | — |
| 15 deg | 0.4599 | 0.938 |
| 30 deg | 0.4076 | 0.977 |
| 60 deg | 0.2998 | 0.992 |
| 90 deg | 0.2576 | 1.000 |
| 180 deg | 0.2790 | 0.984 |

**Win rate is 0.938–1.000 at every angle.** The 11ch v2 model is pose-specific and does not
show the pathology. §12g remains true for *its* checkpoint on *its* 104 PLINDER-panel
pockets; it should not be quoted as a property of the current model, and any plan justified
by it needs re-checking.

## 22b. The reconstruction yardstick, measured properly

`dice(pred, true)` for `poc2mol_v2_11ch_ep576` on the **full HiQBind v2 val split**:

    n = 1019 pockets,  dice = 0.5027 +/- 0.0033 (sem)

Quote that, not §12g's 0.596 (different checkpoint, different pocket set, not comparable to
anything scored on HiQBind val). Beware subsamples: the first 128 pockets give 0.4976 and
the first 256 give 0.5080, a spread of 0.010 that is easy to mistake for a real effect.

⚠️ **`data.config.batch_size` does not control the validation loader.** `ComplexDataModule`
uses `data.val_batch_size` (new, optional; `None` reproduces the old `min(4, batch_size)`).
Any eval script that composes a config and counts *batches* is therefore counting batches of
a size it did not choose. This produced a 128-pocket baseline compared against 256-pocket
numbers and understated a gap by 0.028 before it was caught.

## 22c. The regression model's error mode, characterised

New diagnostics in `scripts/adhoc_analysis/density_diagnostics.py` (model-agnostic; runs on
any protein-grid -> ligand-grid checkpoint). On the full val split:

**Pocket decomposition** — score with the correct pocket, a shuffled (wrong) pocket, and a
zero pocket:

| pocket fed to the model | dice |
|---|---|
| correct | 0.5027 |
| shuffled | 0.2360 |
| zeros (pocket-blind floor) | 0.2073 |

So 0.207 is obtainable knowing nothing, +0.029 comes from having *any* pocket, and **+0.267
from the correct one** — the pocket-specific term dominates, which is the direct evidence
that the density is genuinely conditional rather than a size/composition recital.

**Stray-density amplitude profile** — predicted voxels per pocket where the true grid is
empty: **15,002 above 0.01, 6,374 above 0.05, 1,960 above 0.2, 584 above 0.5**, mean
amplitude 0.101. The 584 confident false positives per pocket are a real characterisation of
how this model errs, and nothing in §1–21 records it.

Why it matters: Dice, MSE and the §18b mass metrics are each blind to a *different* one of
these failure modes. Diffuse low-amplitude error is nearly free under MSE (squaring) but
expensive under mass-based `on_target`/`empty_frac`; concentrated error is the reverse. Two
models can therefore be ranked oppositely by two reasonable metrics, which is exactly what
happened on the generative branch. Report the profile, not just one scalar.

## 22d. Stage-1 decoder scaling has hit diminishing returns

A stage-3 arm run as an experimental control, differing from the earlier one only in the
stage-1 decoder checkpoint:

| stage-1 decoder | steps | stage-3 `val/likelihood_auc_znorm` |
|---|---|---|
| `s1_v2_11ch.ckpt` | 56,912 | 0.7423 |
| `s1_v2_11ch_ep14_step247256.ckpt` | 247,256 | 0.7522 |

**4.3x the stage-1 training bought +0.010 AUC.** Both remain below the parameter-free
composition readout at 0.7615 (§14d). Further stage-1 scaling looks like a poor use of
compute relative to attacking the density or the readout.

## 22e. Operational traps worth not rediscovering

* **`pgrep -f "task_name=X"` matches the shell that runs it**, so `pkill` kills your own
  command mid-script. Collect PIDs into a file first, or put the launch in a script file so
  the pattern is not in the invoking command line. This killed two runs.
* **A launcher health check must assert that a W&B run URL appeared.** A run started with
  `WANDB_MODE=offline` looks identical in the progress bar and cannot be switched to online
  afterwards; 30 minutes of an 8-GPU run was discarded to this.
* **`voxel_aggregation: max` remains untested against `sum`.** It is non-injective
  (separability 0.07, §10b), so atom positions are destroyed before any model sees the
  target, and it caps every downstream metric equally. `sum` with `radius_scale < 1` exists
  in the config and has never been tried.

---

## 23. End-to-end training — the gradient path, and two bugs it exposed (2026-08-12)

Branch `end-to-end`. Goal per CLAUDE.md §0: let the language-modelling loss backpropagate
through Poc2Mol, with supervision at BOTH the voxel and the token layer.

### 23a. The gradient path is open, and verified

`Poc2Mol` ran in `Vox2SmilesDataModule.on_after_batch_transfer` — a datamodule hook outside
the autograd graph, under `torch.no_grad()`, with `_bind` calling `requires_grad_(False)`.
Three independent severances; removing any one alone does nothing.

New code:

| file | role |
|---|---|
| `src/models/end_to_end.py` | `EndToEndPoc2Smiles`, **subclasses `VoxToSmilesModel`** so the likelihood-AUC machinery (pocket×candidate matrix, cross-rank all-gather, column z-norm) is inherited verbatim rather than reimplemented |
| `src/data/vox2smiles/end_to_end.py` | `EndToEndVoxelBuilder` — voxelises only, returns `protein_voxels` / true `ligand_voxels` / `has_pocket`; `pixel_values` is assembled inside the model |
| `configs/model/end_to_end.yaml`, `configs/data/vox2smiles_e2e_v2_11ch.yaml`, `configs/experiment/e2e_*.yaml` | the arms |

Loss is `LM cross-entropy + voxel_loss_weight × BCEDice`, with AdamW param groups so the
upstream carries its own learning rate.

**Measured, not assumed** (smoke test, batch 32, 16 pocket rows):

| condition | Poc2Mol grad norm |
|---|---|
| LM loss only (`voxel_loss_weight = 0`) | **16.6** |
| voxel loss only | 9.5 |
| LM gradient severed, no voxel loss | **exactly 0.0**, all 124 params still in the graph |

So at `voxel_loss_weight = 1.0` the LM term already *outweighs* the reconstruction term
1.7:1. `w = 3.0` is the smallest weight that flips which loss leads, which is why round 2
uses it. Cost: 138 ms/batch, 13.7 GB, ~1.77 s/optimiser step at batch 32 on 2 GPUs.

⚠️ **Freeze by zeroing the TERM, never by detaching.** A detached upstream leaves its
parameters out of the backward pass and plain DDP rejects that outright. `voxel_loss * 0.0`
keeps every parameter in the graph with a zero gradient — a real freeze that DDP accepts.

⚠️ `save_hyperparameters(ignore=[...])` in a subclass does **not** undo the parent's call:
it merges into the existing dict, so the parent's capture of the 117M-parameter
`poc2mol_model` survives and gets pickled into every checkpoint. Pop the key explicitly.

### 23b. 🚨 Stage-3 validation was STOCHASTIC — 0.7522 is not a like-for-like target

`vox2smiles_combined_v2_11ch.yaml` overrides only `use_cluster_member_zero` and `data_path`
on its val datasets, so they inherit `rotate: true, translation: 6.0` from
`complex_dataset_v2_11ch.yaml`. Every stage-3 validation therefore scored a random
augmentation — exactly what §6 forbids for model selection, and what
`poc2mol_hiqbind_v2_11ch.yaml` fixed for Poc2Mol back on 2026-08-06 but was never carried
across to the vox2smiles configs.

Measured cost on the *same* checkpoint: pooled Dice **0.311 augmented vs 0.503
deterministic**. And with deterministic validation the frozen baseline scores **0.7660**,
*above* the published 0.7522 — the augmented validation was simply harder. Comparing new
deterministic arms against 0.7522 would have manufactured a +0.014 win before the
experiment started.

**Consequence:** re-measure the baseline in your own validation regime. That is what arm Z
is for. Do not quote against 0.7522 across a validation change.

### 23c. 🚨 Three incompatible soft-Dice definitions are live in the codebase

On one untouched batch of `poc2mol_v2_11ch_ep576`:

| definition | value | where |
|---|---|---|
| per-sample, pooled over channels **and** space | **0.467** | `density_diagnostics.pooled_soft_dice` — the 0.5027 yardstick |
| per-channel, pooled across the batch, then averaged | 0.311 | the obvious reading of "pooled Dice" |
| per (sample, channel), then averaged | 0.194 | `compute_per_channel_dice`'s shape |

The lower two are dragged down by the ~5 of 11 channels a typical ligand leaves empty, each
contributing an unavoidable zero (§3c). **Always name which one you mean.** With the
yardstick's definition, the frozen arm reads 0.5028 on the full val split against the
documented 0.5027 — an exact independent confirmation that the e2e data path is correct.

### 23d. 🚨 `val_check_interval` counts MICRO-batches — accumulation silently rescales it

`exp1_s3_v2_11ch.yaml` uses `val_check_interval: 250` and is correct, because at
`batch_size 64` on 4 devices its `accumulate_grad_batches` is 1. Round 1 copied that 250
while dropping `batch_size` to 32 (needed for the U-Net backward), which makes
`accumulate_grad_batches = 4` — so validation ran every **62 optimiser steps, not 250**, and
`patience: 12` meant **750 steps, not 3000**.

Result: arm C peaked at step 562 and was stopped at 1312, exactly 750 steps later. Every arm
died between 1312 and 1374 of a nominal 4000, **none reached the LR anneal beginning at step
2000**, and each accumulated ~21 validation draws feeding its "best" value instead of ~5
(inflating the max-selection bias §6 warns about). Round 1 is an early-training snapshot
only.

`src/train.py` now emits a warning whenever `val_check_interval` is an int and
`accumulate_grad_batches > 1`, spelling out the true cadence and the true patience in steps.

### 23e. Round 1 result (TRUNCATED — read as an early-training snapshot)

Ladder: Z frozen → D unfrozen on voxel loss only → B + LM gradient → C anchor loosened to
0.1. All at `poc2mol_lr 1e-4`, effective batch 256, deterministic validation.

| arm | best AUC | @step | Dice | contrast |
|---|---|---|---|---|
| Z frozen | **0.7660** | 937 | 0.5028 | baseline |
| D control | 0.7586 | 1187 | 0.5017 | D−Z = −0.0074 |
| B balanced | 0.7658 | 874 | 0.4806 | **B−D = +0.0073** |
| C lm-dominant | 0.7483 | 562 | 0.4908 | C−B = −0.0176 |

Every contrast is inside the ±0.02 noise band, so **nothing here is a result**. The one
suggestive pattern, and the reason round 2 tightens rather than loosens the anchor: Dice
fell in all three arms whose upstream could move, without a compensating AUC gain, and the
loosest-anchor arm was the clearest loser. Nothing beat the frozen baseline.

**Round 2** (running, W&B `uxufh43w` / `1e3ptob7` / `0d5u8ufc` / `1l41jx1d`) fixes the
cadence and replaces C with `voxel_loss_weight 3.0`. Analysis:
`scripts/adhoc_analysis/e2e_sweep_report.py`.

### 23f. 🚨 EarlyStopping on a noisy metric kills arms at the wrong step

Round 2 fixed the cadence (§23d) and set `patience: 8` = 2000 optimiser steps, which sounds
generous. It is not, because **EarlyStopping monitors the RAW metric**, and the raw metric's
per-check sigma is 0.012 pooled — 0.016 on the arms whose upstream moves.

Arm B spiked to 0.7619 at step **499** while its smoothed level was ~0.747, i.e. roughly a
+2σ draw in its second validation. That spike became the bar, nothing beat it, and the arm
was stopped at step 2499 of 4000 — **before the LR anneal running 2000→4000 had finished**,
the phase §19d found worth +15% Tanimoto. The other three were on track to stop at 2999,
2749 and 3499. Every arm stopping at a different step, chosen by where its noise spike landed,
is not a comparison.

**Round 3 disables early stopping** (`patience: 1000` against 16 checks) so every arm runs
the full 4000 steps, and raises `save_top_k` to 5 because top-k also ranks on the raw metric
— its top 2 are likely to be spikes rather than the genuinely best model.

### 23g. Read SMOOTHED peaks, never raw maxima

Adjacent-check swings inside a single arm reach 0.045 (arm A went 0.7541 → 0.7094 in one
check). With sigma ≈ 0.012 and ~16 draws, the expected maximum sits about
`sigma*sqrt(2*ln(16))` ≈ **0.029** above the true level — larger than any contrast in this
experiment. Ranking arms by "best AUC" therefore ranks luck.

It changes conclusions, not just decimals. Mid-round-2, on raw bests, arm A led and the
frozen baseline was third; on smoothed peaks the frozen baseline led and every arm with a
moving upstream sat below it. `e2e_sweep_report.py` now reports the peak of a centred
3-check rolling mean as the headline, estimates per-check sigma from successive differences,
and prints the implied selection inflation next to the raw best.

Mid-round-2 smoothed standings (step ~1750/4000, so NOT final):

| arm | smoothed peak | raw best | sigma/check |
|---|---|---|---|
| Z frozen | **0.7595** | 0.7609 | 0.0096 |
| B balanced | 0.7475 | 0.7619 | 0.0162 |
| A anchored 3.0 | 0.7431 | 0.7541 | 0.0153 |
| D control | 0.7377 | 0.7430 | 0.0080 |

Contrasts against a ~0.020 floor: unfreezing = **−0.0218** (the only one that clears it, and
it is negative), LM gradient = +0.0098, anchor 1.0→3.0 = −0.0043.

**Also note the noise itself is a result:** arms whose upstream moves carry roughly twice the
per-check sigma of the frozen ones. A non-stationary decoder input shows up as validation
noise, which both makes these arms harder to measure and is a cost of end-to-end training in
its own right.

### 23h. FINAL RESULT — end-to-end does not beat the frozen baseline

Thirteen runs over nine configurations, all at 4000 steps with deterministic validation, all
read as the peak of a 3-check rolling mean. Two seeds where shown as a mean.

| arm | what changed from the baseline | AUC | Dice | seeds |
|---|---|---|---|---|
| **I** | B's density at step 500, frozen | **0.7638** | 0.4967 | 2 |
| **Z** | *the baseline* — frozen upstream, = stage 3 | **0.7610** | 0.5028 | 2 |
| H | B's density at step 1000, frozen | 0.7599 | 0.5006 | 2 |
| B | LM gradient + voxel loss, weight 1.0 | 0.7523 | 0.5122 | 2 |
| F | B's density at step 4000, frozen | 0.7504 | 0.5122 | 2 |
| C | anchor loosened to 0.1 | 0.7483 | 0.4908 | 1 |
| E | upstream lr 1e-5 | 0.7465 | 0.5031 | 1 |
| G | D's density at step 4000, frozen | 0.7463 | 0.5068 | 1 |
| D | upstream trains on voxel loss only | 0.7450 | 0.5068 | 1 |
| A | anchor tightened to 3.0 | 0.7431 | 0.4938 | 1 |

Pooled per-run seed sigma over six replicate pairs = **0.0038**, so a contrast needs ~0.0075
at two seeds per arm and ~0.0107 at one.

**The only contrast that clears it is `F − Z = −0.0106`**: a density given 4000 steps of
end-to-end training reconstructs BEST of anything tested (Dice 0.5122 vs the baseline's
0.5028) and decodes WORST (0.7504 vs 0.7610). Both upstreams frozen and stationary when
measured, same decoder init, so this is a clean statement about the density itself. It is
§8/§14d made concrete: **reconstruction quality and decoder-usable signal are different
axes, and past some point they are actively opposed.**

Everything else is under threshold, including the two results this branch was built to find:

* `B − Z = −0.0087` — end-to-end training is *not* better than leaving Poc2Mol frozen, and
  the direction is negative in every single arm that trained the upstream.
* `F − G = +0.0041` — the LM gradient's isolated contribution to density quality, measured
  with the moving-target confound removed from both sides. Positive, plausibly real, not
  established. This is the number a future attempt should power properly.

**What was eliminated along the way**, each by its own arm rather than by argument: density
collapse into a private code (Dice *rises*, it does not collapse); too weak an anchor
(A − B = −0.0043); too strong an anchor (C − B = +0.0052 at one seed, also null); and the
upstream moving too fast (E − B = −0.0058). The moving-target explanation survived all of
those and is supported by the noise: every arm whose upstream moved carries ~2x the
per-check validation sigma of a frozen one. But removing the moving target entirely, which
is what F/G/H/I do, still does not produce a win.

**The one loose end.** Arm I — 500 steps of end-to-end, then freeze and retrain the decoder
— is the only configuration anywhere above the baseline, at +0.0028. That is 0.37x the
threshold, so it is not a result. It is worth one more seed rather than a shrug: both its
pairs are tight (0.0016, 0.0017 spread, against 0.009 for F and G), and at that sigma three
to four seeds would settle it either way.

**Do not read the drift curve as monotonic.** At one seed it looked like a clean inversion
(AUC falling as Dice rose across four points); at two seeds H drops below Z despite lower
Dice, and the ordering breaks. The honest version is the two-point one: heavy drift is worse,
low drift is indistinguishable from no drift.

---

## 24. Ensembling + mass/decoder fusion — 0.7883, the best number in the project (2026-08-12)

`scripts/adhoc_analysis/fusion_ensemble.py`. 6 end-to-end checkpoints x 4 augmentation
replicates = 24 members, 104-pocket PLINDER panel, 105 candidates, deterministic pocket
selection with rotation ON for the augmentation axis. Members are COLUMN z-normalised before
averaging or blending, matching the metric.

Checkpoints span arms Z (original density), I (drift-500 density) and H (drift-1000 density),
two seeds each — so the member set is diverse in decoder weights AND in upstream density,
which §20 found is the diversity that pays.

| | decoder | composition |
|---|---|---|
| single member | 0.7367 (sd 0.0143) | 0.7246 (sd 0.0182) |
| + augmentation (x4) | 0.7546 (+0.0179) | 0.7360 (+0.0114) |
| + checkpoint (x6) | 0.7758 (+0.0391) | 0.7393 (+0.0147) |
| + both (24) | **0.7818** (+0.0451) | 0.7408 (+0.0162) |

**FUSION of the two fully-ensembled readouts** — the axis §20 named as untested:

| w_composition | 0.0 | 0.1 | **0.2** | 0.3 | 0.4 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|---|
| AUC | 0.7818 | 0.7863 | **0.7883** | 0.7881 | 0.7858 | 0.7850 | 0.7408 |

**0.7883 against a single deterministic checkpoint's 0.7612 on the same panel: +0.027.**

Reading, in order of how much each axis bought:

* **Checkpoint ensembling is the big one, +0.039.** Consistent with §20's +0.030 on the
  composition readout alone.
* **Augmentation is worth +0.018 alone but only +0.006 on top of checkpoints** (0.7758 ->
  0.7818). Strongly sub-additive, exactly as §20 measured (+0.0023 there). Do checkpoint
  ensembling first; augmentation is what you reach for when you have one model.
* **Fusion adds +0.0065 on top of the full ensemble**, and the optimum is at
  w_composition = 0.2, not the 50/50 of §12d.

⚠️ **The two readouts are much LESS orthogonal here than §12d measured.** Within-pocket
`corr(decoder, composition)` = **+0.547**, against +0.024 on the 9ch/v1 pipeline. That is the
mechanism behind both differences above: more correlated readouts mean a smaller fusion gain
(+0.0065 here vs +0.024 there) and an optimum weighted towards the stronger member rather
than 50/50. Do not carry §12d's r = +0.024 across pipelines.

⚠️ **The composition readout is much weaker here than the decoder** (0.7408 vs 0.7818
ensembled), where §12a had it *stronger* (0.761 vs 0.722). Note §12b's finding that S, Cl, Br
and I carry no signal at all — with the 11-channel scheme the readout has more dead channels
to average over, which is a plausible cause and is worth checking before assuming the readout
is simply worse.

⚠️ **w = 0.2 was chosen on the panel it is scored on**, so 0.7883 carries a little selection
optimism. The curve is flat though — anything in w = 0.1-0.3 gives >= 0.786 — so the effect
is not a knife-edge, and quoting 0.786 for a pre-committed w = 0.2 would be the conservative
statement.

**Comparison with §12d's 0.785:** that figure was ROW-standardised while the metric
z-normalises by column, which §20 measured as ~+0.005 of inflation, so its like-for-like value
is ~0.780. 0.7883 here is computed with the correct normalisation throughout.
