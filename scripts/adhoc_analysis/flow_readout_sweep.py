"""Find the best DICE readout of a trained flow model, without retraining anything.

`dice(pred, true)` is maximised in expectation by the conditional MEAN, which is precisely
what the regression Poc2Mol is trained to output (0.5027 on this split). A single draw from
a generative model is therefore structurally penalised -- it commits to one plausible answer
where the metric rewards hedging. That is a property of the metric, not a failure of the
model, and it is fixable at inference time: a flow model can emit the mean directly.

This sweeps every readout that estimates the mean, plus the knobs that narrow the sampled
distribution, and reports Dice for each against the regression baseline:

    sample        one full ODE trajectory (what val/sample/dice logs)
    mean_of_k     average of k complete samples -- Monte-Carlo estimate of the mean
    one_step      x0 + v(x0, 0, c) = E[x1 | x0, c], ONE forward pass, averaged over k draws
    one_step_mid  the same question asked at t = 0.5

crossed with guidance scale and noise temperature (< 1 shrinks x0, narrowing the
distribution being averaged over).

Usage:
    ./venvPlixer/bin/python scripts/adhoc_analysis/flow_readout_sweep.py \\
        --ckpt <path> --n_batches 4 --batch_size 16 --output /tmp/readout.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

from flow_sampling_watchdog import build_batches, load_flow_checkpoint  # noqa: E402
from poc2mol_rotation_control import soft_dice  # noqa: E402

REGRESSION_YARDSTICK = 0.5027


def score(pred, true):
    return soft_dice(pred.float(), true.float())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default="hiqbind", choices=["hiqbind", "zinc"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_batches", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, step = load_flow_checkpoint(args.ckpt, device)
    batches = build_batches(args.data, args.batch_size, args.n_batches, device)
    n_pockets = sum(l.shape[0] for _, l in batches)
    print(f"{args.ckpt}\n  global_step {step}, {n_pockets} pockets from {args.data} val\n")

    # (label, kwargs for predict_expected or None for plain sampling)
    trials = [("sample (k=1)", None, dict(guidance_scale=1.0)),
              ("sample (k=1)", None, dict(guidance_scale=2.0)),
              ("sample (k=1)", None, dict(guidance_scale=3.0))]
    for w in (1.0, 2.0, 3.0):
        for k in (4, 16):
            trials.append((f"one_step k={k}", "one_step", dict(guidance_scale=w, n_draws=k)))
        trials.append((f"mean_of_k k=4", "mean_of_k", dict(guidance_scale=w, n_draws=4)))
    for temp in (0.0, 0.5):
        trials.append((f"one_step k=1 T={temp}", "one_step",
                       dict(guidance_scale=2.0, n_draws=1, temperature=temp)))
    trials.append(("one_step_mid k=4", "one_step_mid", dict(guidance_scale=2.0, n_draws=4)))

    header = f"{'readout':>22} {'guid':>5} {'dice':>8} {'sem':>7} {'vs 0.5027':>11}"
    print(header); print("-" * len(header))
    results = {}
    for label, mode, kw in trials:
        w = kw.get("guidance_scale", 1.0)
        dices = []
        for b, (protein, ligand) in enumerate(batches):
            gen = torch.Generator(device=device); gen.manual_seed(args.seed + b)
            with torch.no_grad():
                if mode is None:
                    pred = model.sample(protein=protein, n_steps=args.steps,
                                        guidance_scale=w, generator=gen)
                else:
                    pred = model.predict_expected(
                        protein=protein, mode=mode, n_steps=args.steps,
                        generator=gen, **{k: v for k, v in kw.items()})
            dices.append(score(pred, ligand))
        d = np.concatenate(dices)
        sem = float(d.std(ddof=1) / np.sqrt(d.size))
        key = f"{label} g={w}"
        results[key] = {"dice": float(d.mean()), "sem": sem, "n": int(d.size)}
        flag = "  BEATS IT" if d.mean() > REGRESSION_YARDSTICK else ""
        print(f"{label:>22} {w:>5.1f} {d.mean():>8.4f} {sem:>7.4f} "
              f"{d.mean() - REGRESSION_YARDSTICK:>+11.4f}{flag}")

    best = max(results.items(), key=lambda kv: kv[1]["dice"])
    print(f"\nbest readout: {best[0]}  dice {best[1]['dice']:.4f} "
          f"(regression baseline {REGRESSION_YARDSTICK})")
    if args.output:
        with open(args.output, "w") as fh:
            json.dump({"ckpt": args.ckpt, "step": step, "results": results,
                       "best": best[0], "yardstick": REGRESSION_YARDSTICK}, fh, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
