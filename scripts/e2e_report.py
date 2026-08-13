"""Read both end-to-end sweeps out of W&B and print them as one table.

The cold ladder (group ``e2e_v1``, nebius2, decoder from stage 1) and the warm ladder
(group ``e2e_warm_v1``, nebius1, decoder from stage 3) are running concurrently against the
same metric, the same data and the same code, so they belong in one table. Run:

    ./venvPlixer/bin/python scripts/e2e_report.py

Three columns matter together, and reading the first alone will mislead:

``AUC znorm``  the objective. Compare each arm to its OWN ladder's frozen baseline (arm Z
               for cold, arm W0 for warm), NOT to the published 0.7522/0.7759 -- those came
               from stochastic-validation runs and carry CLAUDE.md §6's ~+0.025
               maximum-selection inflation. Both sweeps validate deterministically.
``dice``       Poc2Mol's own reconstruction quality, same definition as
               density_diagnostics.py, so directly comparable to the 0.5027 yardstick. An
               AUC win with a collapsed Dice means the upstream has stopped being a density
               model and become a private code for the decoder.
``zinc loss``  the forgetting tripwire. The decoder can buy pocket performance by giving up
               its ZINC ability, and this turns first.

`best` is a maximum over validation checks, so it carries selection bias of its own; the
`@step` column is there to show whether an arm peaked early and decayed (overfitting) or is
still climbing (undertrained, raise max_steps).
"""

from __future__ import annotations

import sys

import wandb

ENTITY_PROJECT = "cath/voxelSmiles"
GROUPS = {
    "cold (stage-1 decoder, nebius2)": "e2e_v1",
    "warm (stage-3 decoder, nebius1)": "e2e_warm_v1",
}
KEYS = [
    "trainer/global_step",
    "val/likelihood_auc_znorm",
    "val/poc2mol/dice/dataloader_idx_1",
    "val/poc2mol/loss",
    "val/zinc/loss",
    "val/poc2mol/tanimoto",
]


#  A run that died before it had a few validation checks carries no usable peak, only a
#  single noisy reading that sorts into the middle of the table and reads like a result. The
#  ENOSPC incident on 2026-08-12 left seven such stubs. Runs still RUNNING are always shown,
#  however few checks they have -- that is live progress, not a corpse.
MIN_CHECKS = 3


def rows_for(api, group):
    out = []
    for run in api.runs(ENTITY_PROJECT, filters={"group": group}):
        history = run.history(keys=KEYS, pandas=False)
        auc = [h for h in history if h.get("val/likelihood_auc_znorm") is not None]
        if not auc:
            if run.state == "running":
                out.append({"name": run.name, "state": run.state, "auc": None})
            continue
        if run.state != "running" and len(auc) < MIN_CHECKS:
            continue
        best = max(auc, key=lambda h: h["val/likelihood_auc_znorm"])
        last = auc[-1]
        out.append({
            "name": run.name,
            "state": run.state,
            "auc": best["val/likelihood_auc_znorm"],
            "step": best.get("trainer/global_step"),
            "last_auc": last["val/likelihood_auc_znorm"],
            "last_step": last.get("trainer/global_step"),
            "checks": len(auc),
            "dice": best.get("val/poc2mol/dice/dataloader_idx_1"),
            "zinc": best.get("val/zinc/loss"),
            "tanimoto": best.get("val/poc2mol/tanimoto"),
        })
    out.sort(key=lambda r: -(r["auc"] or -1))
    return out


def fmt(value, spec=">8.4f"):
    return format(value, spec) if isinstance(value, (int, float)) else f"{'--':>8}"


def main():
    api = wandb.Api()
    header = (f"{'run':30s} {'state':>9} {'best AUC':>9} {'@step':>6} "
              f"{'last AUC':>9} {'@step':>6} {'dice':>8} {'zinc CE':>8} {'tanimoto':>9}")
    for title, group in GROUPS.items():
        print(f"\n=== {title}   [group {group}]")
        print(header)
        print("-" * len(header))
        try:
            rows = rows_for(api, group)
        except Exception as error:  # a missing group is normal before that sweep starts
            print(f"  could not read: {error}")
            continue
        if not rows:
            print("  no runs yet")
        for r in rows:
            print(f"{r['name'][:30]:30s} {r['state']:>9} {fmt(r.get('auc'), '>9.4f')} "
                  f"{str(r.get('step', '--')):>6} {fmt(r.get('last_auc'), '>9.4f')} "
                  f"{str(r.get('last_step', '--')):>6} {fmt(r.get('dice'))} "
                  f"{fmt(r.get('zinc'))} {fmt(r.get('tanimoto'), '>9.4f')}")

    print("\nRead each arm against its own ladder's frozen baseline (Z cold / W0 warm).")
    print("Poc2Mol reconstruction yardstick: dice 0.5027. Composition-only readout: AUC 0.7615.")
    print("0.7522 / 0.7759 are STOCHASTIC-validation figures and are not like-for-like here.")


if __name__ == "__main__":
    sys.exit(main())
