# Extending the Plixer test sets — data survey and plan (2026-08-06)

Motivation: the strict (non-chronological) test splits are underpowered — **107 PLINDER and 141
seq-sim systems** against 943 chronological. This document scopes what data has become available
since the training data was compiled, and what a leakage-controlled replacement should look like.

The framing follows Mattsson & Walters, *Identifying and Addressing Systematic Data Leakage in
Protein-Ligand Affinity Benchmarks*, bioRxiv [10.64898/2026.06.29.735309](https://doi.org/10.64898/2026.06.29.735309)
(NTAB; [code](https://github.com/bamattsson/ntab), [data](https://zenodo.org/records/19665374)).

Numbers below are reproducible with `scripts/adhoc_analysis/paper_audit_new_pdb_pool.py`.

---

## 1. What NTAB actually argues, and which parts bind on us

| NTAB claim | Applies to Plixer? |
|---|---|
| Protein sequence-identity splits are insufficient — "target mirroring" means homologues with low overall identity have correlated binding profiles; >6,000 such assay pairs in ChEMBL 36, and leakage persists down to **0.2** sequence identity | **Yes.** Our strict split uses a single 30% MMseqs2 threshold. NTAB's Fig. 5 says >4,000 correlated assay pairs survive even at 0.5. |
| A single similarity threshold is the wrong instrument; **bin the test set into ligand-novelty tiers** and report per tier | **Yes.** Directly replaces our binary "strict subset" with a graded axis. |
| Benchmarks must be validated against a **ligand-only / pocket-blind baseline** — if it matches the structural model, the benchmark is leaky | **Yes — and we are in unusually good shape here.** See §2. |
| Temporal split alone is insufficient (best ligand-only model still reaches r = 0.32) | **Yes.** Our chronological split is a temporal split. |
| Their specific critique of Boltz-2 / IsoDDE claims on FEP+4 and OpenFE | Indirect — but it means those benchmarks are poor headline choices for us too. |

**The mechanism of leakage is different for us, and this matters.** NTAB's leakage is
*assay-value* mirroring across homologous targets in ChEMBL. Plixer never trains on ChEMBL,
BindingDB, or any affinity data — it trains on HiQBind co-crystal geometry and ZINC20 ligand
conformers. So target mirroring in the NTAB sense cannot reach us. Our exposure is
**pocket→ligand co-occurrence** memorised from PDB structures, which is the Runs N' Poses
critique rather than the NTAB one. Both should be controlled; they are not the same axis.

### Note on ZINC20
ZINC20 (5.66M ligands) trains the Vox2Smiles decoder only, and carries no pocket labels, so it
cannot leak the pocket→ligand association that the hit/decoy and enrichment metrics test. It is
therefore **not** treated as a leakage source here. The one residual effect — decoder familiarity
with a test ligand's SMILES raising its likelihood — is largely symmetric, because decoys are drawn
from the same test pool. (§5.2 of `CLAUDE.md` has the measured numbers if this needs revisiting.)

---

## 2. The asset we should be leading with

NTAB's central diagnostic is the pocket-blind baseline. From `CLAUDE.md` §5.1, we already ran it:

> a pocket-blind baseline scoring by ligand mean likelihood alone gets **AUC exactly 0.500**

That is the control NTAB demands, and it is the one Boltz-2 and IsoDDE fail (r = 0.66 and 0.36 from
a ligand-only model on FEP+4 / OpenFE). Our ranking signal is, by construction, entirely in the
pocket×ligand interaction term — the variance decomposition puts 84% of raw likelihood variance in
the ligand main effect, but that term contributes *exactly zero* to the ranking metric.

This should be stated explicitly and early in the thesis chapter. It is a stronger claim than the
current evaluation makes for itself, and it is already measured.

---

## 3. The training cutoff, established from PDB release dates

Measured over all 17,650 PDB entries in the HiQBind parquet splits:

| split | systems | PDB entries | release date range |
|---|---|---|---|
| train | 25,365 | 14,575 | 1982-05-26 → **2019-12-25** |
| val | 3,385 | 1,643 | 1991-10-15 → **2019-12-18** |
| test | 2,681 | 1,444 | **2020-01-01** → 2022-01-12 |

Two things follow:

1. **The chronological split is clean** — no train/val entry is released on or after 2020-01-01.
2. **HiQBind was compiled around January 2022** (test: 1,109 entries in 2020, 321 in 2021, 13 in
   2022, none after 2022-01-12). Everything in the PDB since then is unseen by *any* Plixer stage.

The model's true structural training cutoff is **2019-12-25**. This is worth stating precisely,
because it makes several external benchmarks zero-shot for us that are not zero-shot for the
co-folding models they were designed to test (Runs N' Poses assumes a 2021-09-30 cutoff).

---

## 4. Size of the post-cutoff pool

RCSB search API, entries released on/after 2022-01-13:

| window | protein entries | + drug-like ligand (MW 150–800) | + resolution ≤ 2.5 Å |
|---|---|---|---|
| **all post-cutoff** | **70,947** | **40,660** | **24,268** |
| 2022 | 14,054 | 8,266 | 5,245 |
| 2023 | 14,216 | 8,232 | 5,207 |
| 2024 | 15,251 | 8,509 | 4,921 |
| 2025 | 17,247 | 10,076 | 5,819 |
| 2026 YTD | 10,565 | 5,812 | 3,234 |

The MW filter is loose — it admits buffers, cryoprotectants and sugars — so 24,268 is an upper
bound before real ligand-quality curation. Expect substantial attrition.

### Ligand chemotype novelty (the number that matters)

Chemical components first released after the cutoff, MW 150–800, non-polymer: **13,269**. For
comparison, the *entire* Plixer train+val ligand vocabulary is **11,743 CCDs / 12,068 unique SMILES**.
More novel chemotype has entered the PDB since the cutoff than the model has ever seen.

Binning those 13,189 (of 13,269, RDKit-parseable) by max Morgan-Tanimoto to train+val ligands,
using NTAB's tiers:

| tier | n | share |
|---|---|---|
| **[0.00, 0.35)** — NTAB's hardest, "novel binder discovery" | **6,581** | **49.9%** |
| [0.35, 0.50) | 4,655 | 35.3% |
| [0.50, 0.70) | 1,605 | 12.2% |
| [0.70, 1.00) | 330 | 2.5% |
| = 1.00 (exact match) | 18 | 0.1% |

Median max-similarity 0.350.

**Roughly 6,500 ligand chemotypes sit in the strictest novelty tier.** Even after curation losses
and protein-level filtering, this replaces a 107-system strict split with something one to two
orders of magnitude larger. The power problem is solvable with data that already exists.

---

## 5. Candidate sources

### 5a. Structural test sets (no affinity needed)

Our two headline metrics — similarity-enrichment and hit/decoy likelihood AUC — need a pocket and a
known ligand, **not** an affinity value. This lifts the binding-data requirement that constrains
HiQBind and widens the usable pool considerably.

| Source | What it gives | Status for us |
|---|---|---|
| **Re-run HiQBind-WF on post-2022 PDB** ([code](https://github.com/THGLab/HiQBind)) | A test set curated by *identical* rules to the training data — no curation-induced distribution shift | **Preferred primary route.** Only source that removes curation as a confound. Requires affinity data (BindingDB/MOAD/BioLiP2) so it is the most filtered option. |
| **Runs N' Poses** ([repo](https://github.com/plinder-org/runs-n-poses), [Zenodo v6](https://zenodo.org/records/18366081), Nat Struct Mol Biol 2026) | ~2,600 high-resolution complexes, pre-annotated with pocket similarity, SuCOS-pocket cluster ID, Morgan/topological Tanimoto to training data | **Best off-the-shelf option.** Its similarity annotations are exactly the pocket-level axis NTAB says we are missing. Zero-shot for us (their assumed cutoff 2021-09-30 is *later* than our 2019-12-25). |
| **PLINDER** ([site](https://www.plinder.sh/)) | 449,383 systems, protein/pocket/interaction/ligand similarity metrics, **98,473 apo + 205,300 AF2 structures linked to holo** | Already used for our current strict split. Built from PDB as of 2024-04-09, so it now covers most of the new pool. The apo/predicted pairings enable §6a. |
| Direct RCSB pull + own curation | Maximum coverage (24k entries) | Fallback; loses curation comparability. |

### 5a-i. Runs N' Poses: the cofactor concern, quantified

Measured from `annotations.csv` (Zenodo 18366081 v6): 4,235 ligand rows / 2,579 systems / 2,565 PDB
entries. `ligand_is_proper` excludes ions and crystallisation artifacts (1,209 rows, 28.5%) but
**not cofactors**. Of the 3,026 "proper" rows:

| | n | share |
|---|---|---|
| cofactor / nucleotide / metabolite CCD | 710 | **23.5%** |
| MW outside 150–800 | 95 | 3.1% |
| zero rings | 134 | 4.4% |
| **drug-like remainder** | **2,179** | **72.0%** |

Top "proper" ligands are HEM (124), FAD (58), ADP (47), ANP (40), NAP (37), H4B (37), NAD (36),
ATP (34), FMN (30). For contrast, HiQBind is **3.2%** cofactor by the same CCD list — a 7× difference.
The concern is real and is confirmed.

It is also one filter away. After removing cofactors + MW 150–800 + ≥1 ring: **2,179 rows /
2,113 PDB entries / 2,006 unique ligand CCDs** — already ~20× the current 107-system strict split.
Using RnP's own `morgan_tanimoto` annotation, **55.6% of that remainder sits in NTAB's hardest
`[0, 0.35)` tier**, closely matching the 49.9% measured independently in §4.

### 5a-i-b. BUILT: the filtered Runs N' Poses evaluation set (2026-08-06)

`scripts/create_runs_n_poses_eval_set.py` → `../runs_n_poses/parquet/test/`, in the layout
`ParquetDataset` loads directly. Two filtering stages, each reporting marginal drops.

**Stage 1 — annotation-based** (on Runs N' Poses' own columns):

| step | dropped | remaining |
|---|---|---|
| annotations | — | 4,235 |
| `ligand_is_proper` (ions, artifacts) | 1,209 | 3,026 |
| cofactor / metabolite CCD blocklist | 725 | 2,301 |
| MW outside [150, 800] | 73 | 2,228 |
| fewer than 1 ring | 64 | 2,164 |
| fewer than 10 heavy atoms | 2 | 2,162 |

**Stage 2 — structural** (on the parsed ligand, added after visual inspection showed the CCD
blocklist leaking ~19% non-drug-like content — IMP, dNTPs, glycans, fragments):

| step | dropped |
|---|---|
| peptide / oligomer CCD code (hyphenated) | 12 |
| nucleoside / nucleotide (N-glycosidic bond to a furanose) | 152 |
| free sugar (sugar ring + ≥2 OH, no aromatic) | 54 |
| QED < 0.15 | 43 |
| MW < 200 | 40 |
| oligosaccharide (≥2 sugar rings) | 25 |
| | **→ 1,836** |

**1,836 systems / 1,800 PDB entries / 1,715 unique ligand CCDs**, 614 pocket clusters,
1,458 ligand clusters, 1,524 protein–ligand cluster pairs. Compare the current strict splits:
107 (PLINDER) and 141 (seq-sim). Every system postdates the 2019-12-25 training cutoff.

Verified zero residual: nucleosides, oligosaccharides, free sugars, peptides, sub-200 Da fragments
all at 0%. QED median 0.553 → **0.594**, q10 0.234 → 0.313, MW median 370.

NTAB novelty tiers vs HiQBind train+val ligands (stored per row as `max_tanimoto_to_train` and
`novelty_tier`, so metrics can be reported stratified):

| tier | n | share |
|---|---|---|
| [0.00, 0.35) | 779 | 42.4% |
| [0.35, 0.50) | 618 | 33.7% |
| [0.50, 0.70) | 271 | 14.8% |
| [0.70, 1.00) | 95 | 5.2% |
| exact match | 73 | 4.0% |

Tightening *raised* the hardest-tier share (38.1% → 42.4%): cofactors and nucleotides recur in both
train and test, so they were concentrated in the high-similarity tiers. The hardest tier alone
(779 systems) is ~7× the entire current PLINDER strict split. The 73 exact matches are ligands whose
SMILES already appear in HiQBind train — retained deliberately, since NTAB's whole argument is that
these should be *reported separately* rather than silently dropped.

Two deliberate non-filters, both verified by eye on the contact sheets
(`scripts/adhoc_analysis/plot_rnp_ligand_sample.py`): **C-nucleosides** (immucillin/forodesine-like)
and **carbasugars** (voglibose-like glycosidase inhibitors) survive, because the patterns key on an
N-glycosidic bond and a ring oxygen respectively. Both are genuine drugs, so chasing them would cost
more than it gains. 78 phosphorus-containing ligands (4.2%) are kept on purpose — phosphinic-acid
pseudopeptides are legitimate designed inhibitors, so a blanket "contains P" rule would be wrong.

Verified end-to-end: loads through `ParquetDataset` and voxelises through `VoxelBatchBuilder` with
no empty grids, and channel occupancies track HiQBind test closely (protein C/O/N/S means
0.340/0.116/0.107/0.013 vs 0.334/0.104/0.102/0.005 — the sulfur gap is hydrogen dilution, not a
parsing difference; see below). Ligands are notably more halogen-rich than HiQBind (F 1.57% vs
0.75%, Cl 0.98% vs 0.30% of ligand atoms), which is useful given §3b bug 2 left those channels
permanently empty in the old training data.

> ⚠️ **Hydrogens.** HiQBind proteins are pdbfixer-protonated (49.8% H); Runs N' Poses receptors are
> raw deposited coordinates (5.7% H) and its ligand SDFs are heavy-atom only. Harmless under the
> 4-channel protein / 9-channel ligand encoding — hydrogen maps to no channel — but **it invalidates
> the `h5` and `hall6` arms of sweep 2 (§3e)**, which give hydrogen its own protein channel. Those
> would need protonated receptors (pdbfixer, i.e. the HiQBind conda env) before they can be scored
> on this set.

Open follow-ups for this set: the cofactor list is a hand-curated CCD blocklist (`--keep-cofactors`
reproduces the unfiltered set); protein-level novelty against HiQBind train still needs MMseqs2,
which is **not currently installed on this node**.

### 5a-ii. Re-running HiQBind-WF: feasibility

**The 2022-01-12 cutoff has a single identifiable cause**, in `pre_process/create_hiqbind_input.ipynb`:

- cell 5 fetches `BindingMOAD_data_for_PDB_1_31_2022.csv` — a **frozen 2022-01-31 snapshot**;
- cell 9 iterates `for _, row in moad.iterrows()`, so **Binding MOAD drives ligand selection**;
- 22,891 of 31,572 HiQBind entries (**72.5%**) take their affinity from MOAD.

Binding MOAD updates only every few years, so the notebook as shipped cannot be extended. **This is
a data-source problem, not a compute problem.**

The fix: **BioLiP2** is current (version 2026-06-29, updated weekly) and already aggregates
MOAD (25,977) + BindingDB (19,330) + an LLM-mined set (23,502) + manual (81) = **68,890 entries with
binding affinity**. It carries every column the selection loop needs (`PDBID`, `Ligand CCD`,
`Ligand chain`, `Ligand residue sequence number`, four affinity columns). So the rewrite is confined
to `pre_process/`; `process.py` — the part that does the actual structural work — is untouched.

**Filters `process.py` actually applies** (exact constants, lines 40–61):

| constant | value | effect |
|---|---|---|
| `MAX_HEAVY_ATOMS` | 4 | misleadingly named — ligands with **fewer than** 4 heavy atoms are discarded |
| `STERIC_CLASH_THRESH` | 2.0 Å | ligand–protein contact below this → entry discarded |
| `BINDING_CUTOFF` | 10.0 Å | protein chains within 10 Å of the ligand are retained |
| `HETATM_CUTOFF` | 4.0 Å | HETATMs within 4 Å of a retained chain are kept |
| `MIN/MAX_NUM_RES_POLY` | 2 / 20 | chains of 2–20 residues are treated as peptide ligands |
| `MAX_ADD_MISSING_RES` | 10 | gaps longer than this are not rebuilt |
| ion/water blocklist | ~60 CCDs | includes `HOH`; rare elements in a ligand also cause discard |

Plus `LigandFixer` (bond orders and protonation against the RCSB reference SMILES, via a modified
`dimorphite_dl`) and `ProteinFixer` (pdbfixer; missing atoms and residues). PDB IDs the workflow
rejects are recorded in `error_fix/` — 1,400 small-molecule and 830 polymer entries for HiQBind.

Selection filters, in the part that would be rewritten: affinity must be Kd/Ki/IC50/EC50 (preference
Kd > Ki > IC50 > EC50) from one of the four sources, with a `logvalue > 3` sanity rejection; ions are
stripped before deciding small-molecule vs polymer.

> ⚠️ **HiQBind applies no drug-likeness filter.** It records QED in the metadata but never filters on
> it. The reason it is 3.2% cofactor while Runs N' Poses is 23.5% is that **the affinity requirement
> acts as a de facto drug-likeness filter** — nobody reports a Kd for the HEM in most structures.
> Consequence: since Plixer's metrics need only pocket + ligand identity and *not* affinity, dropping
> the affinity requirement would greatly increase yield — but it would also remove the implicit
> cofactor filter, so an explicit one (QED / cofactor blocklist) would have to replace it.

**Cost estimate.** The README quotes ~1 day on a 256-core CPU for the full 39,399-row input, i.e.
~9 core-minutes per row. A post-cutoff input of order 10–20k rows is ~100 core-days. This node has
128 cores but is shared with the training runs, so budget 48–64 cores → **~2–4 days wall clock**,
plus mmCIF downloads. Add ~1–2 days to rewrite `pre_process/` against BioLiP2, and a separate conda
env (`env.yml`: gemmi, openmm, openff-toolkit, pdbfixer, openbabel — *not* venvPlixer). **≈1 week
of effort overall.**

### 5b. Affinity data

| Source | Contents | Fit |
|---|---|---|
| **NTAB** (ChEMBL 36) | Time-split, novelty-tiered, assay-filtered affinity measurements + target sequences | We never train on ChEMBL, so **the whole of NTAB — including its train split — is test data for us**. Caveat: NTAB has no structures; targets must be mapped UniProt→PDB holo or AF2 model. And novelty tiers must be **recomputed against HiQBind train**, not NTAB's own train split. |
| **OpenBind release 1** ([blog](https://openbind.uk/news/blog-openbinds-first-release-a-structure-affinity-dataset-for-structure-based-ai/), Zenodo 10.5281/zenodo.20026661) | 925 crystallographic binding events, 699 compounds, KD for 601 (494 after QC), EV-A71 / CVA16 2A protease, **CC0** | **Excellent focused case study.** Structures *and* affinities, single target, released May 2026 — entirely post-cutoff, and generated after our model existed, so effectively prospective. Small (one target) so it cannot carry the headline. |
| **CASP16** ligand category | 140 affinity targets over 5 systems, blind at the time | Useful, small. Best CASP16 affinity Kendall τ = 0.42 sets a realistic bar. |
| **Polaris / ASAP** antiviral challenge | SARS-CoV-2 + MERS Mpro, chronological split, potency + pose; 965 complexes | Good secondary; real lead-optimisation data. |
| **FEP+4 / OpenFE** | Congeneric series, structures + ΔG | **Use only as a stress test, not a headline** — see §7. |
| **BindingDB / LIT-PCBA / DUD-E** | Local-only (`/mnt/disk2/VoxelDiffOuter/...`), not on this node | LIT-PCBA remains the best real actives/inactives source (`CLAUDE.md` §6). |
| **CACHE** | Prospective, experimentally tested selections | Not a retrospective benchmark; a future direction. |

---

## 6. Benchmarking strategies worth adding

### 6a. Apo and AlphaFold pockets
PLINDER already ships apo and AF2 structures linked to their holo counterparts, so this needs no new
data generation. It tests the thing that actually determines prospective usefulness: in a real
campaign you do not have the holo structure. Poc2Mol is already trained with ±6 Å random
translation, i.e. explicitly for pocket mis-specification, so we have a prior reason to expect it to
degrade gracefully — and a measurable claim if it does. Report holo / apo / AF2 side by side.

### 6b. Affinity correlation instead of binary hit-vs-decoy
Replace (or supplement) the hit/decoy AUC with Spearman correlation between the pocket-conditional
score and pKi/pIC50 within a target. This requires the **likelihood-ratio scorer**
(`CLAUDE.md` §8 open item 4): `log P(S | pocket) − log P(S | pocket-free reference)`, using the
ZINC-only decoder as the reference. The raw likelihood cannot do this — 84% of its variance is
ligand-intrinsic and correlates −0.758 with heavy-atom count, so a raw-likelihood correlation would
mostly measure molecular size. The z-normalisation trick works for ranking but needs a pocket panel;
the likelihood ratio is the pointwise mutual information and is the deployable form.

### 6c. Novelty-tiered reporting throughout
Report every metric stratified by max ligand Tanimoto to HiQBind train, in NTAB's bins
`[0, 0.35) / [0.35, 0.5) / [0.5, 0.7) / [0.7, 1.0) / = 1.0`, **and** by a pocket-level similarity
tier from Runs N' Poses or PLINDER. Two axes, not one threshold. This subsumes the current
"strict split" and turns `CLAUDE.md` §5.2's seen/unseen stratification into a proper graded result.

### 6d. Virtual-screening metrics
Replace global ROC-AUC with **EF@1%, EF@5%, BEDROC (α=20)** (`CLAUDE.md` §5.3), and report ligand
efficiency (Vina/heavy-atom count) wherever Vina appears.

---

## 7. Honest caveats

1. **FEP benchmarks are a poor fit for Plixer's scoring metric.** Congeneric series are the worst
   case: ligands within a series are near-identical, so pocket-conditional likelihood has almost no
   dynamic range to work with, and the ligand-intrinsic term dominates what range there is. Expect
   weak correlations. Worth running as an honest negative/stress result, not as a headline.
2. **The §3b voxelisation fixes mean released and newly trained checkpoints are not comparable.**
   Any new benchmark table must be generated from one checkpoint under one encoding.
3. **13,269 new CCDs is a chemotype count, not a system count**, and is measured before any
   ligand-quality curation. The usable figure after HiQBind-style filtering will be materially lower.
4. **Protein-side novelty is not yet measured.** The tiering in §4 is ligand-only. The equivalent
   protein/pocket-level audit against HiQBind train needs an MMseqs2 run plus pocket comparison — that
   is the main missing piece of this survey.
5. **`evaluation_results/`, LIT-PCBA and DUD-E are not on this node** (local-only, 48 GB). Anything
   depending on them has to run locally or be copied over.

---

## 8. Suggested order of work

1. Measure protein/pocket-level novelty of the post-cutoff pool against HiQBind train (MMseqs2 at
   several thresholds — 0.9 / 0.5 / 0.3 / 0.2 — to reproduce NTAB's Fig. 5 argument on *our* data).
2. Pull **Runs N' Poses**, apply `ligand_is_proper` **plus an explicit cofactor/drug-likeness
   filter** (→ ~2,100 systems), and evaluate the current checkpoint on it. Hours of work; fastest
   path to a real number on a leakage-controlled set, and its pocket-similarity annotations are
   already the right ones.
3. Re-run **HiQBind-WF** on post-2022 PDB as the curation-matched primary set. Requires rewriting
   `pre_process/` to be BioLiP2-driven first — see §5a-ii. ~1 week.
4. Implement the **likelihood-ratio scorer**, verify it reproduces the z-normalised AUCs
   (0.854 / 0.753 / 0.782), then use it for affinity correlation.
5. Evaluate on **OpenBind** EV-A71 as a post-hoc-prospective case study.
6. Add **apo / AF2** pocket variants via PLINDER pairings.
7. Regenerate all tables from a single checkpoint, reported on both novelty axes.
