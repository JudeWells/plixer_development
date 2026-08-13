# Plixer — working brief

> ⚠️ **Working rules for this branch.** Work **only** on the `end-to-end-2` branch — do not
> switch branches and do not create new ones. **Committing on this branch is allowed** (as of
> 2026-08-13); do not push, and do not commit on any other branch.
>
> Commits matter here for a concrete reason, not tidiness: `src/utils/provenance.py` stamps every
> checkpoint with a git commit and writes an `uncommitted.patch`, but that patch only captures
> **tracked-but-modified** files. While the RL work was uncommitted, 81 untracked `.py`/`.yaml`
> files — including `src/models/rl_vox2smiles.py`, the whole DPO implementation — were recorded
> nowhere, so `commit + patch` could not reconstruct any run. Keep new source committed or the
> provenance embedded in every checkpoint is decorative.

Full measured history is in **`report.md`** (§1–22) — consult it for detail; this file is the
brief. §22 carries corrections and baselines back from the `generative-poc2mol` branch and
should be read before trusting §12g or any Dice figure below.

Two-stage pocket-conditioned molecule generator: **Poc2Mol** (3D U-Net, protein voxels → ligand
voxels) → **Vox2Smiles** (ViT encoder + GPT-2 decoder, ligand voxels → SMILES).
Context: PhD thesis chapter. Paper: `plixer_ICML_GenBio_2025_version.pdf`.

---

## 0. CURRENT TASK — end-to-end training (branch `end-to-end-2`, this node)

**Goal: let the language-modelling loss backprop all the way through Poc2Mol**, so the density
is optimised for what the decoder actually needs rather than for reconstruction. Target:
`val/likelihood_auc_znorm` above **0.77**.

⚠️ **Two sweeps are running, on two nodes, from two branches.** They are complementary; do not
duplicate one on the other node.

| node | branch | sweep | decoder starts from |
|---|---|---|---|
| nebius2 | `end-to-end` | `e2e_z_frozen` / `e2e_d_control` / `e2e_b_balanced` / `e2e_c_lm_dominant` | stage 1, `s1_v2_11ch_ep14_step247256.ckpt` |
| **nebius1** | **`end-to-end-2`** | `e2e_w0_warm_frozen` / `e2e_w1_warm_lr1e5` / `e2e_w2_warm_lr1e4` / `e2e_w3_warm_declr2e5` | **stage 3**, `checkpoints/s3_v2_11ch/step_0003000_auc_0.7576.ckpt` |

The implementation (`src/models/end_to_end.py`, `src/data/vox2smiles/end_to_end.py`, the
`end_to_end: true` branch in `Vox2SmilesDataModule`) was written on the `end-to-end` branch and
copied here unchanged, so the two sweeps' numbers are directly comparable. Only the experiment
configs and `scripts/launchers/launch_e2e_warm_sweep.sh` are new here.

**Keeping the node busy unattended.** `scripts/launchers/autofill_gpus.sh` watches for free GPU
pairs and launches the next arm from `scripts/launchers/queue.txt` (one experiment name per line;
launched lines are rewritten in place with a `# launched HH:MM:SS` prefix, so the file is also the
record of what ran and in what order — append to extend the plan). It refuses to launch below
25 GB of disk. `watch_gpu_idle.sh` is the passive companion that only reports.

⚠️ **Two failure modes cost four dead launches before this worked; both are silent.**
- **Port reuse.** The DDP rendezvous port must be chosen by *availability* (`ss -Hltn`), never an
  in-process counter — a counter resets when the script restarts and reissues ports that running
  jobs hold. Torch then dies at `init_process_group` with "Address already in use", but the
  launcher sees a successful background, so the GPUs return to looking idle and the *next* queued
  arm is launched onto them too. A whole queue can drain into nothing this way.
- **Startup blindness.** A run spends ~2 min indexing ZINC before allocating GPU memory, so
  `nvidia-smi` reports its GPUs free that entire time. An in-process cooldown does not survive a
  restart of the filler, so restarting it mid-startup double-books the pair. Fixed with a
  **persistent claims file** (`logs/.gpu_claims`: gpu, pid, timestamp) — a GPU counts as busy if
  claimed by a live pid inside the grace window even when nvidia-smi disagrees. On disk, because
  that is what makes it restart-safe.

⚠️ **Do not hand-launch while the filler is running.** It owns the free-GPU→launch path; a manual
launch races it (this happened, two jobs onto one pair four seconds apart). To run something
sooner, put it at the FRONT of the queue instead.

**Why the gradient path needed surgery.** Poc2Mol ran in `Poc2MolInferenceBuilder`, called from
`Vox2SmilesDataModule.on_after_batch_transfer` — a *datamodule* hook, outside the autograd graph
the LightningModule builds. It was severed three further times: `torch.no_grad()`, `_bind`'s
`.eval()`, and `requires_grad_(False)` on every parameter. Deleting any one alone does nothing.
`EndToEndPoc2Smiles` (a `VoxToSmilesModel` subclass, so `val/likelihood_auc_znorm` is computed by
literally the same code) owns Poc2Mol and runs it inside `training_step`.

### Fusion ensembles — 0.8080 against the packaged bundle's 0.7883

Scored by `scripts/adhoc_analysis/fusion_ensemble.py`, copied **verbatim** from
`../plixer_ensemble_20260812/code/` so the number comes from that bundle's protocol rather than a
reimplementation. Same 104×105 PLINDER panel; alignment (panel, positive mask, valid columns)
verified identical across all three matrix sets before combining.

| set | ckpts | decoder | composition | fused | @w | corr(dec,comp) |
|---|---|---|---|---|---|---|
| bundle (e2e/frozen) | 6 | 0.7818 | 0.7408 | 0.7883 | 0.2 | 0.547 |
| **D — DPO, AUC-selected** | 6 | 0.7651 | 0.7445 | **0.7984** | 0.5 | 0.431 |
| **A — AUC-selected** | 4 | 0.7733 | **0.7582** | **0.8057** | 0.5 | **0.399** |
| B — DPO, tanimoto-selected | 6 | 0.7541 | 0.7445 | 0.7871 | 0.5 | 0.469 |
| **bundle + A** | 10 | 0.7978 | 0.7476 | **0.8080** | 0.3 | |
| **bundle + A + D** | 16 | 0.7930 | 0.7475 | **0.8090** | 0.3 | |

**bundle+A wins at EVERY weight 0.1–0.6, including 0.8076 at the bundle's own pre-committed
w = 0.2**, so the gain is not the blend weight moving to suit our readouts. Paired bootstrap over
pockets (400 resamples): **+0.0199 vs the bundle, 95% CI [+0.0068, +0.0333], excludes zero**.

**The gain is member quality, not member count.** A alone (4 checkpoints) scores 0.8057; adding
the bundle's six gives +0.0022, and adding D as well only reaches 0.8090 (+0.0032 over A), both
with CIs spanning zero. Pooling the *tanimoto*-selected B set instead makes things **worse**
(bundle+A+B = 0.8050 against bundle+A = 0.8080) — AUC-weak members dilute rather than diversify.
A's edge is a stronger composition readout (0.7582 vs 0.7408) that is markedly less correlated
with the decoder (0.399 vs 0.547), which is what lets fusion contribute +0.032 instead of +0.0065. Two of A's four members carry end-to-end-drifted
Poc2Mol upstreams, which is the density-diversity axis the bundle's §5 identified.

**A DPO-only ensemble beats the reference at matched member count: 0.7984 vs 0.7883 (6 vs 6),**
and its members are individually stronger (mean AUC 0.7653 against the bundle's 0.7367) while also
far ahead on tanimoto — DPO is better on both axes at the MODEL level. ⚠️ But the win is
weight-dependent: at the bundle's pre-committed w = 0.2 the DPO ensemble is *behind* (0.7832 vs
0.7883), because its more orthogonal readouts want w ≈ 0.5. A (mixed density) wins at every weight;
D does not.

🔑 **DPO is capped on the composition axis by construction.** RL freezes Poc2Mol, so all six DPO
members share one density and their composition matrices are near-duplicates: that ensemble reaches
0.7445, essentially the bundle's 0.7408, against 0.7582 for A's two density families. The decoder
axis and the composition axis need different kinds of diversity — DPO supplies the first, distinct
upstreams the second, and fusion needs both. `rl_dpo_auc_upstep500` / `_upstep1000` train DPO
decoders on the bundle's drifted upstreams to get both in one member.

⚠️ **B is the informative negative-ish result:** six DPO checkpoints selected on *tanimoto* —
i.e. ~0.018 per member below their own AUC peak — still fuse to 0.7871, level with the bundle.
The DPO family carries real diversity; those members were simply picked at the wrong step.
`rl_dpo_auc_*` rebuilds them at the AUC peak.

⚠️ The bootstrap covers **panel-sampling noise only**, not training-seed variance, so it is a
lower bound on the uncertainty. No ensemble here has been rebuilt from independent seeds.

Reproduce: `scripts/adhoc_analysis/combine_fusion_members.py --sets name=path.npz ...` — pools
saved member matrices in numpy, no GPU. Matrices in `results/e2e/fusion_{aucsel4,dpotan6}.npz`.

### ⚠️ Provenance of checkpoints written before 2026-08-13 06:35

Every checkpoint carries an embedded provenance record, but runs before the commits below stamp
`2f10f95-dirty` with 81 untracked `.py`/`.yaml` files — including `src/models/rl_vox2smiles.py`
— captured in neither the commit nor `uncommitted.patch` (which only records tracked-but-modified
files). Those runs are **not** reconstructable from their stamped commit alone. The equivalent
source is now committed:

| commit | contents |
|---|---|
| `4ebc809` | `end_to_end.py`, `rl_vox2smiles.py`, the datamodule hook, the diversity metrics |
| `a891763` | every experiment / model / data config |
| `f0e4cad` | analysis tooling, launchers, results, this record |

Training-code mtimes are all ≤ 05:50 on 2026-08-13 while the `rl_dpo_auc_*` members started
06:32, so **those members ran byte-identical source to `4ebc809`+`a891763`** despite their record
saying dirty. For anything earlier, the resolved config in the run dir is authoritative (as in the
`plixer_ensemble_20260812` bundle's §4), and `uncommitted.patch` covers the three tracked files.

**Best end-to-end checkpoints so far** (nebius1; each carries BOTH models, hence 2.4 GB):

| path | AUC | Dice | note |
|---|---|---|---|
| `checkpoints/e2e_best/dpo_s45_tan0.1495.ckpt` | ~0.759 | 0.5028 | **the best model here** — DPO, tanimoto 0.1495 |
| `checkpoints/e2e_best/frozen_pipeline_s49_auc0.7657.ckpt` | 0.7657 | 0.5028 | best pure-SFT pipeline |
| `checkpoints/e2e_best/e2e_w4_anneal_step500_auc0.7718.ckpt` | 0.7718 | 0.4992 | end-to-end; density near-intact |
| `checkpoints/e2e_best/e2e_w2_warm_step312_auc0.7759.ckpt` | 0.7759 | 0.4870 | end-to-end; degraded density, steep decay |

The frozen one is recommended despite the lower number: end-to-end was measured at −0.0015 against
it (n.s.) with 4× the run-to-run spread and a degraded density. ⚠️ 0.7657 is the **max over 8
seeds** scored on the same 104-pocket panel it was selected on, so it is optimistically biased by
roughly the +0.025 §6 describes; **0.7589 ± 0.0059 is the honest expected performance** of this
recipe. Ensemble member matrices and summary are preserved in `results/e2e/ens8.{json,npz}`, so
`ensemble_subset_search.py` can be run without repeating any forward passes.

⚠️ **Both are upper-tail draws, not demonstrated improvements** — the same config reruns at
0.7330–0.7779 across seeds (see the seed table below), and the frozen pipeline's mean matches
theirs. They are kept because they are the best *checkpoints on disk*, not because end-to-end
training was shown to produce them. Do not present either as a method win.

Run checkpoints under `logs/e2e_*/` are deleted once a round is read; W&B keeps every metric, so
only the weights are lost, and only for arms that lost. Promote a keeper to `checkpoints/e2e_best/`
**before** trimming — the disk footgun below is why this is a routine and not an afterthought.

**Read arms against the right baseline, not against a published number.** Both sweeps validate
*deterministically*; every stage-3 run before them validated with `rotate: true, translation: 6.0`
inherited from `complex_dataset_v2_11ch.yaml`, i.e. **stochastically**, against §6. So 0.7522 and
0.7759 are not like-for-like targets and carry §6's ~+0.025 maximum-selection inflation. `e2e_z_frozen`
re-measures the frozen baseline for the cold ladder; `e2e_w0_warm_frozen` does it for the warm one.

**An AUC win bought by destroying the density is not a win.** Watch `val/poc2mol/dice` against
**0.5027** throughout — it uses the same definition as `density_diagnostics.py`, so the two are
directly comparable. If it collapses while the AUC climbs, Poc2Mol has stopped being a density
model and become a private code for the decoder: no longer diagnosable by any existing tool.

### Measured so far (2026-08-12, deterministic validation, warm ladder)

| arm | decoder lr | poc2mol lr | best AUC | @step | Dice at best |
|---|---|---|---|---|---|
| W0 frozen upstream | 5e-5 | — | 0.7575 | 249 | 0.5028 |
| W1 | 5e-5 | 1e-5 | 0.7507 | 124 | 0.5031 |
| W2 | 5e-5 | 1e-4 | 0.7759 | 312 | 0.4870 |
| W3 | 2e-5 | 1e-4 | 0.7620 | 499 | 0.4947 |

### 🚨 SETTLED: the end-to-end gradient does nothing, and costs stability

**Paired 8-vs-8, seeds 42–49.** `e2e_w4_anneal` (end-to-end) against `e2e_w12_frozen_anneal` —
identical in every respect except the three switches that sever the upstream
(`voxel_loss_weight: 0`, `poc2mol_lr: 0`, `lm_grad_to_poc2mol: false`):

| seed | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | mean | σ |
|---|---|---|---|---|---|---|---|---|---|---|
| end-to-end | .7718 | .7330 | .7471 | .7779 | .7563 | .7711 | .7502 | .7447 | **0.7565** | 0.0157 |
| frozen | .7616 | .7507 | .7594 | .7575 | .7625 | .7641 | .7499 | .7657 | **0.7589** | **0.0059** |
| diff | +.0102 | −.0177 | −.0123 | +.0204 | −.0062 | +.0070 | +.0003 | −.0210 | **−0.0024** | 0.0145 |

**Paired difference −0.0024 ± 0.0051 (SEM), t = −0.47 on 7 df, 95% CI [−0.0145, +0.0097].**
Null. Note the CI's *upper* bound is +0.0097 — even the optimistic end of the interval is below
the ~0.010 effect this setup can resolve, so this is a genuine null, not merely underpowered.

**The variance inflation IS significant.** σ 0.0157 against 0.0059 is a variance ratio of 7.1×,
F = 7.07 on (7,7) df against a 4.99 critical value (p < 0.05). End-to-end training makes the
pipeline reliably *less* reproducible while leaving the mean alone.

σ 0.0059 (frozen) is the number to size any future experiment against: an effect below ~0.01 is
not detectable here without many seeds.

**The second finding is the more useful one: end-to-end training multiplies the variance by 20×**
(σ 0.0210 against 0.0047) while leaving the mean untouched. Unfreezing Poc2Mol does not buy
performance; it buys instability. The 0.7759/0.7779 figures that made this look like a win are
the upper tail of that inflated spread — and 0.7330 is the same experiment's lower tail.

⚠️ **The metric is NOT the noisy thing — this experiment is.** The frozen pipeline scores
0.7507–0.7616 (σ 0.0047) on the *same* 104-pocket panel, so `val/likelihood_auc_znorm` resolves
differences of ~0.01 perfectly well. An earlier guess that the 104-pocket panel drove the spread
is contradicted by this: widening the panel would not have helped.

### RL on the decoder (GRPO / DPO, Tanimoto reward) — this one WORKS

`src/models/rl_vox2smiles.py`, configs `rl_grpo` / `rl_dpo` / `rl_control`. Poc2Mol frozen; the
decoder is the policy. Sample `group_size` molecules per pocket, score each by Morgan-Tanimoto
to that pocket's true ligand, and either standardise the rewards within the group as a GRPO
advantage or reduce the group to a best-vs-worst DPO pair. Both anchored to a frozen snapshot
of the starting policy (GRPO by explicit KL, DPO by its reference terms).

All arms at 4 seeds (42–45), tanimoto measured at n=200:

| arm | tanimoto | AUC at that step |
|---|---|---|
| **DPO** | 0.1482 ± 0.0011 | **0.7585 ± 0.0006** |
| SFT comparator | 0.1465 ± **0.0028** | 0.7433 ± 0.0057 |
| GRPO | 0.1444 ± 0.0012 | 0.7589 ± 0.0009 |
| control, `lr: 0` | 0.1372 ± **0.0000** | 0.7522 |

**On tanimoto alone, nothing beats SFT.** DPO − SFT = +0.0017 ± 0.0015 (t = 1.16, n.s.);
GRPO − SFT = −0.0021. SFT's best seed (0.1500) is the highest tanimoto of any arm in the
experiment. Detecting a gap this small against SFT's σ = 0.0028 needs ~42 seeds per arm, so it is
not cheaply establishable and is not worth chasing.

**What DOES hold, strongly: DPO reaches the same tanimoto while KEEPING the ranking metric.**
AUC 0.7585 against SFT's 0.7433 — **+0.0152, t = 5.27**, complete seed separation. SFT buys its
tanimoto late in the run, at a step where its AUC has decayed to 0.735–0.747; DPO gets there
without paying. On the joint objective DPO dominates, and that is the claim to make. GRPO also
preserves AUC but gains nothing on tanimoto, so it is dominated by SFT and DPO both.

`val/zinc/loss` stayed flat at 0.0132–0.0133 for every RL arm: the KL leash held and the
calibration collapse usually expected from RL did not appear.

### Constant-LR sweep: the schedule was the binding constraint, not the method

Constant LR (50-step warmup then flat), 3000 steps, early stopping unreachable. Tanimoto at n=200;
AUC and ZINC read at the tanimoto peak. SFT comparator is 0.1430.

| lr | best tanimoto | @step | AUC | ZINC CE |
|---|---|---|---|---|
| 2e-6 | 0.1496 | 2749 | 0.7566 | 0.0132 |
| 5e-6 (the old default) | 0.1540 / 0.1565 | 2799 / 2499 | 0.761 / 0.760 | 0.0136 |
| 1e-5 | 0.1662 / 0.1726 | 1899 / 2949 | 0.755 / 0.749 | 0.0144 / 0.0152 |
| 1.5e-5 | 0.1727 | 2849 | 0.7450 | 0.0146 |
| 2e-5 | **0.1787** | 2849 *(still rising)* | 0.7460 | **0.0188** |

⚠️ **Seed spread at fixed LR is σ ≈ 0.0046** (1e-5: 0.1662 and 0.1726). That is 4× the σ the
annealed runs suggested, and it sets what the sweep can resolve: 5e-6 → 1e-5 (+0.014, 3σ) is
real; 1e-5 → 2e-5 (+0.009, 2σ) is suggestive but rests on ONE seed at 2e-5; 1e-5 → 1.5e-5
(+0.003) is noise. `rl_dpo_const_lr2e5_s43` replicates the top setting before it is believed.

**The trade-off is monotonic in LR and is the real story.** Tanimoto rises 0.1496 → 0.1787 while
ZINC CE rises 0.0132 → 0.0188 and AUC falls 0.761 → 0.746, without exception across seven arms.
This is a Pareto frontier, not a single optimum: "best LR" is only meaningful once it is decided
how much pocket-conditioning and ZINC ability a tanimoto point is worth.

**DPO's advantage over SFT is +0.036, not the +0.0071 the annealed runs suggested.** Every earlier
figure was produced by a schedule chosen before anything about the optimum was known: 400 steps
for something that wants ≥2800, at a quarter of the useful LR, annealing before it stopped
improving. `rl_dpo.yaml`'s 5e-6 / 400 steps is simply the wrong recipe — start from 2e-5 and
≥3000 constant steps.

⚠️ **Do NOT quote 0.1787 yet.** ZINC cross-entropy rose 42% (0.0132 → 0.0188) and AUC fell to
0.7460 at that setting. Those are exactly what the KL penalty exists to prevent, and `kl_beta`
was tuned at 5e-6 — 4× lower. Two readings are still open: the larger step genuinely finds a
better pocket-conditioned policy, OR it escapes a leash calibrated for a smaller step and part of
the gain is drift toward the unconditional HiQBind scaffold distribution, which is the degenerate
optimum of a Tanimoto reward and needs no pocket at all. `rl_dpo_lr2e5_kb005` / `_kb01` raise
`kl_beta` at fixed LR to separate them: tanimoto holding near 0.1787 while ZINC and AUC recover
means the gain is real; tanimoto collapsing when the leash tightens means it was drift.

🚨 **"DPO CONVERGES TO A PLATEAU" IS RETRACTED — it may be the LR schedule, not the model.**
The 1200-step arms use warmup 50 / stable 550 / decay 600, so the LR falls from 5e-6 at step 600
to 2.5e-7 at 1200 (68% of peak by step 800, 37% by 1000). A converged model and a model whose
learning rate was annealed away produce the SAME flat curve, and the very tight second-half
spread (sd 0.0008) cited as evidence of convergence is precisely what a decaying LR manufactures.
`val/likelihood_auc_znorm` was still RISING when the decay began, which is the tell.

Early stopping compounded it: seed 42 was killed at step **999**, not 1200, for 8 checks without
improvement — but it was at 37% of peak LR by then, so "not improving" was partly by construction.

The completed decayed set makes the point sharply. Peaks 0.1509 / 0.1534 / 0.1514 at steps
549 / 799 / 549 — all at or just past where decay begins (600) — and end-of-run values of
0.1500 / 0.1502 / 0.1502, **sd 0.0001**. Three seeds agreeing to the fourth decimal is not
convergence of the model, it is the learning rate going to zero underneath all three. Two of the
three were also cut short by early stopping (steps 999 and 949) rather than reaching max_steps.

`rl_dpo_const` / `rl_dpo_const_s43` re-run this at **constant 5e-6, 3000 steps, early stopping
unreachable**, to measure where the curve actually turns over. **Set `max_steps` and the decay
position from that measurement**; every DPO number above was produced by a schedule chosen before
anyone knew where the optimum was.

⚠️ **THE RL ARMS WERE TRUNCATED, NOT CONVERGED — the table above understates DPO.** Three of four
DPO seeds have their maximum in the last one or two validation checks, and the gap between max and
end-of-run is +0.0004 to +0.0022. `max_steps: 400` was chosen by analogy with the AUC arms, which
all peaked early and decayed; the RL arms do the opposite. Extended 1200-step runs reach
0.1487 by step 349 — the 400-step runs' *final* value, with 850 steps still to go.

**Compare on END-OF-RUN value, not max-over-checks.** Peak-picking systematically flatters the
noisier method: SFT gains +0.0016 from selection, DPO only +0.0009, because SFT's curve oscillates
(0.1437, 0.1367, 0.1436, 0.1359, …) while DPO's rises monotonically. On last-3-checks means the
comparison is DPO 0.1473 ± 0.0006 against SFT 0.1449 ± 0.0033, i.e. **+0.0024 ± 0.0017** rather
than +0.0017. SFT's tails are rising too, so `sft_n200_long` extends it to 1600 steps — extending
only the treatment would manufacture the result.

⚠️ **This table's first version claimed DPO − SFT = +0.0038 at p < 0.05.** That used only TWO SFT
seeds, whose σ (0.0011) understated the true spread by 2.5×. Two seeds is not a variance estimate.
The same mistake had already been made once this session on the end-to-end arms; measure the
comparator at the same seed count as the treatment BEFORE reporting a difference.

🔑 **The `lr: 0` control has sd 0.0000 and its maximum is the FIRST validation check.** Validation
is deterministic (`rotate: false, translation: 0`) and `generate_smiles` defaults to greedy, so an
unchanged policy emits byte-identical molecules at every check. **`val/poc2mol/tanimoto` therefore
has no measurement noise at all**, and the max-over-checks inflation that §6 warns about is
exactly zero for it — unlike `likelihood_auc_znorm`, whose σ is 0.0059. That makes tanimoto a far
cheaper metric to run experiments against: a single seed is meaningful, and differences of ~0.003
are readable. Always include an `lr: 0` arm to confirm it, since the property depends on the val
augmentation staying off.

⚠️ **`n_samples_for_validity_testing` changes WHICH POCKETS are scored, not just how many.** It
takes the first n rows of the first validation batch, so n=30 scores the first 30 pockets and
n=200 the first 200 — a different and evidently easier subset. The same policy family reads
**0.157 at n=30 and 0.1372 at n=200**. Tanimoto numbers from runs with different `n` are NOT
comparable, which invalidated the first baseline quoted for this experiment. Compare only at
matched n; `sft_n200` re-measures the supervised arms at n=200 for exactly this reason.

⚠️ The reference policy is snapshotted in `on_fit_start`, NOT `__init__` — `src/train.py` applies
`init_weights_from` after `instantiate`, so a reference built in the constructor would be a
randomly-initialised network, the KL would pull towards noise, and DPO's reference terms would be
meaningless, all while logging plausible curves.

### Decoder-seed ensembling also fails, and the reason is diagnostic

`scripts/adhoc_analysis/decoder_ensembling.py` (new; ensembles the **decoder** axis on the
likelihood metric, where `poc2mol_ensembling.py` ensembles the *Poc2Mol* axis and would be a
no-op here since all these arms share one frozen upstream). Eight frozen seeds:

| | value |
|---|---|
| solo znorm AUC, mean of 8 | 0.7589 (sd 0.0061) |
| 8-member ensemble | **0.7641** |
| gain over solo mean | +0.0052 |
| gain over **best** member | **−0.0014** |

The ensemble does not beat its own best member, and +0.0052 is far below §20's +0.030.

**Why: the members are not independent.** Member–member score correlation is **ρ = 0.948**
(min 0.937, max 0.967), which by `N/(1+(N-1)ρ)` is **1.05 effective independent members out of
8**. All eight start from the *same* stage-3 decoder checkpoint and train only 800 steps, so
they are eight small perturbations of one model, differing only in batch order and augmentation.
Averaging near-identical predictors cannot reduce variance.

**Consequence for any future ensembling work:** diversity has to come from the initialisation,
not the seed — different stage-1 checkpoints, different channel schemes, or the near-orthogonal
composition readout (§12d: within-pocket r = +0.024 against the decoder, and their blend reaches
0.785, still the best number in the project). Re-running seeds off a shared warm start is
measuring nothing, and that is probably worth checking against however §20's +0.030 was obtained.

Validation of the new script: the pocket-blind control returns exactly 0.5000 for every member,
and the solo AUCs reproduce the training-logged values to ~0.001 (mean 0.7589 either way), so it
scores the same quantity by the same code path (it reuses the model's own
`_accumulate_likelihood_rows`). ⚠️ It needs `bf16-mixed` autocast to match training numerics —
Poc2Mol's master weights are fp32 while the voxels arrive bf16.

**Practical consequence: ship the frozen pipeline.** It matches end-to-end on the mean, is 4×
tighter run-to-run, and holds Dice at 0.5028 in every seed instead of degrading it to 0.487–0.499.

The cold ladder on nebius2 agrees independently: arm Z (frozen) 0.7660 vs arm B (end-to-end)
0.7658. Two separately-built ladders, same conclusion.

---

**Everything below predates the replication and is kept only as a record of how the null was
reached. Do not quote a number from it.**

The table above reads as "+0.018 for end-to-end over the frozen baseline" (W2 0.7759 vs W0
0.7575). **It does not replicate.** Rerunning ONE configuration -- `e2e_w4_anneal`, unchanged --
at four seeds gives:

| seed | 42 | 43 | 44 | 45 | mean | stdev | range |
|---|---|---|---|---|---|---|---|
| best AUC | 0.7718 | 0.7330 | 0.7471 | 0.7779 | **0.7574** | **0.0210** | 0.0449 |

The frozen baseline is 0.7575. **The difference of means is −0.0000.** The 0.7759 and 0.7779
headline figures are the upper tail of a distribution centred exactly on the frozen pipeline.

Seed-to-seed σ is 0.021 — **larger than every effect chased in rounds 1–3**, so the entire arm
ranking in the tables below is consistent with noise, including W4's apparent advantage over
W5/W6/W7. Treat those orderings as unmeasured, not as findings.

The cold ladder said the same thing before replication was considered: arm Z (frozen) 0.7660
against arm B (end-to-end) 0.7658. Two independent ladders agreeing on no effect is the stronger
statement.

⚠️ **Do not run another arm of this experiment without replicates.** Four rounds of tuning were
spent reading a ±0.021 noise floor as signal. §6 already warned this metric moves ±0.02 between
adjacent checks and that "best val X" carries ~+0.025 maximum-selection inflation; the cost of
ignoring it was the whole afternoon. Runs are ~20 min on 2 GPUs — replication is nearly free and
is not optional.

Remaining gap in the argument: W0 is a single seed and ran the *old* schedule, so the comparison
above is a distribution against a point. Arms **W12/W13/W14** are W4's exact config with only the
three upstream-severing switches flipped (`voxel_loss_weight: 0`, `poc2mol_lr: 0`,
`lm_grad_to_poc2mol: false`), at matched seeds, which settles it properly.

⚠️ **The upstream learning rate does NOT behave as `configs/model/end_to_end.yaml` predicted.**
That comment reasoned a 576-epoch-deep Poc2Mol receiving an unfamiliar gradient wants a rate
well below the 1e-4 it was trained at. Measured, 1e-5 is *worse than not moving at all* (W1
0.7507 < W0 0.7575) while 1e-4 is much better. More upstream movement is better across the
range tested; `poc2mol_lr: 1e-5` is the wrong default.

⚠️ **Every arm in both ladders peaks between step 250 and 900 and then decays**, and Dice decays
with it (W2: 0.5027 → 0.4870).

**Round 2 — resizing the LR schedule to the observed optimum is the fix.** `e2e_w4_anneal`
(50 warmup / 250 stable / 400 decay, `max_steps: 800`, everything else W2's) is the best
configuration found:

| arm | change from W4 | best AUC | @step | Dice |
|---|---|---|---|---|
| **W4 anneal** | — | **0.7718** | 499 | **0.4992** |
| W6 | decoder lr 2e-5 | 0.7636 | 62 | 0.4866 |
| W7 | poc2mol_lr 3e-4 | 0.7576 | 499 | 0.4877 |
| W5 | voxel_loss_weight 3.0 | 0.7534 | 62 | 0.4738 |

W4 matches W2's AUC within noise **while keeping Dice at 0.4992 against the 0.5027 yardstick**,
where W2 degraded it to 0.4870 — and it decays far more gently afterwards (0.7515 at step 749
against W2's 0.7391). The peak also moved from 312 to 499, i.e. into the decay window, which is
the signature of a correctly-sized schedule. **Prefer W4's config over W2's**, despite W2 holding
the nominally higher single number.

Every direction away from W4 lost: upstream rate above 1e-4 as well as below it, a lower decoder
rate, and a heavier density anchor. `voxel_loss_weight: 3.0` is notable — the density anchor was
the obvious lever for the Dice problem and it **hurt both metrics**; the schedule fix solved the
same problem instead.

🚨 **This round-2 ranking did not survive replication** — see the seed table above. W4's four
seeds span 0.7330–0.7779, so the 0.7718 in this table is one draw and the gaps to W5/W6/W7
(0.018–0.018) are smaller than the 0.021 spread of a single configuration. **The only claim here
that still stands is the mechanical one**: resizing the schedule moved the peak from step 312
into the decay window at ~500, which is a shape change visible in every arm rather than a metric
difference. Whether it buys AUC or Dice is unmeasured.

**Footguns specific to this work** (beyond §6):
- ⚠️ **An end-to-end checkpoint is 2.4 GB**, not the ~1 GB a decoder-only one is — it carries
  both models plus AdamW state for 289M parameters. At the inherited `save_top_k: 2` +
  `save_last: True` that is **7 GB per arm**, and six concurrent arms filled the 193 GB disk in
  45 minutes. ENOSPC killed four runs outright and took down the two long-running ones when
  their checkpoint writes failed — including one arm mid-ascent. The anneal configs set
  `save_top_k: 1, save_last: False`; `scripts/launchers/watch_gpu_idle.sh` now warns below
  30 GB. Check `df` before launching more than four arms.
- ⚠️ **`val_check_interval` counts BATCHES, not optimiser steps**, so it is divided by
  `accumulate_grad_batches` — which `src/train.py` computes, so it changes with world size.
  At `devices: 2` (accumulate 4) `val_check_interval: 250` validates every **62** steps, and
  `EarlyStopping(patience=12)` therefore allows 750 steps, not the 3000 the stage-3 config
  intended. Runs were stopping at ~900 steps with the LR decay scheduled to begin at 2000, so
  no arm ever saw its anneal — §19d again, and it cost this sweep a round.
- `quality_filter` blanks a rejected row's labels, so that row contributes no LM loss and
  therefore **no gradient to Poc2Mol** — exactly the samples it most needs. Keep it `none`.
- Half the mixture is ZINC (`prob_poc2mol: 0.5`) and carries no pocket, so the effective batch
  for the *upstream* is half what it looks like.
- A batch with no pocket rows leaves every Poc2Mol parameter out of the graph, which plain DDP
  rejects outright. `build_pixel_values` handles it by zeroing the *term*, never detaching.
- `_bind` hard-casts Poc2Mol to bfloat16; that is right for frozen inference and wrong here, so
  `EndToEndPoc2Smiles.__init__` forces it back to fp32 and lets autocast handle the forward.
- `batch_size` is 32, not stage 3's 64: backprop now spans the 117M-param U-Net as well as the
  172M decoder. `target_samples_per_batch: 256` keeps the effective batch matched regardless.

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
