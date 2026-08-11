"""Hyperparameter sweep for the stage-3 decoder fine-tune (9-channel, no protein channels).

Target metric is `val/likelihood_auc_znorm`, which the A/B baseline topped out at 0.7759 — only
just above the parameter-free composition readout's 0.7615 (CLAUDE.md 12a), so there is a concrete
bar to clear.

Why the runs are short: `val/poc2mol/loss` bottoms at ~step 1300 (~37k Poc2Mol samples seen) and
everything after is overfitting on HiQBind's 9,872 clusters. Early stopping on that metric turns a
fixed step budget into "run until it turns", which both saves time and removes `max_steps` as a
confound between configs.

Axes were chosen from what the diagnostics say is binding:
  lr             -- the usual suspect, and it interacts with how fast overfitting arrives
  weight_decay   -- NEVER deliberately set before today; every prior run used AdamW's 0.01
  dropout        -- the ViT encoder runs at 0.0 (the GPT-2 decoder already defaults to 0.1)
  prob_poc2mol   -- lower means each HiQBind cluster is revisited less often per step, and more
                    ZINC anchoring. The released combined model used 0.3 (CLAUDE.md 4b).

Usage:
    python scripts/sweep_s3.py --hours 6 [--dry_run]
Results:
    python scripts/sweep_s3.py --report --group <wandb group>
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import time
from datetime import datetime, timedelta

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]

FIXED = [
    "experiment=exp1_s3_baseline",
    "init_weights_from=checkpoints/s1_maxagg_last_9ch.ckpt",
    "trainer.devices=2",
    "data.num_workers=12",
    "data.config.batch_size=150",
    "data.config.target_samples_per_batch=300",
    "data.config.voxel_aggregation=max",
    "data.predicted_ramp_start_step=200",
    "data.predicted_ramp_end_step=2000",
    "model.n_samples_for_validity_testing=50",
    "model.config.scheduler.num_warmup_steps=200",
    "model.config.scheduler.num_stable_steps=1800",
    "model.config.scheduler.num_decay_steps=2000",
    "model.config.scheduler.min_lr_ratio=0.03",
    "+trainer.max_steps=4000",
    "trainer.max_epochs=10000",
    "trainer.val_check_interval=200",
    "+trainer.limit_val_batches=10",
    "+trainer.num_sanity_val_steps=0",
    # the sweep needs metrics, not weights; 3 full checkpoints per run would be ~3 GB each
    "callbacks.model_checkpoint.save_top_k=1",
    "callbacks.model_checkpoint.save_last=False",
    "seed=42",
]

GRID = {
    "model.config.lr": [2e-5, 5e-5, 1e-4],
    "model.config.weight_decay": [0.01, 0.1],
    "dropout": [0.0, 0.1],          # expands to both ViT dropout keys
    "data.train_dataset.prob_poc2mol": [0.25, 0.5],
}


def build_configs():
    keys = list(GRID)
    configs = []
    for values in itertools.product(*(GRID[k] for k in keys)):
        overrides, tags = [], []
        for key, value in zip(keys, values):
            if key == "dropout":
                overrides += [f"model.config.hidden_dropout_prob={value}",
                              f"model.config.attention_probs_dropout_prob={value}"]
                tags.append(f"do{value}")
            else:
                overrides.append(f"{key}={value}")
                short = {"model.config.lr": "lr", "model.config.weight_decay": "wd",
                         "data.train_dataset.prob_poc2mol": "pp"}[key]
                tags.append(f"{short}{value}")
        configs.append({"name": "_".join(tags), "overrides": overrides})
    return configs


def launch(config, gpus, group, log_dir):
    name = config["name"]
    overrides = FIXED + config["overrides"] + [
        f"task_name=sweep_{name}",
        f"logger.wandb.group={group}",
        f"logger.wandb.name={name}",
    ]
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=",".join(str(g) for g in gpus),
               PROJECT_ROOT=REPO, TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    log_path = os.path.join(log_dir, f"{name}.log")
    handle = open(log_path, "w")
    process = subprocess.Popen(
        [os.path.join(REPO, "venvPlixer/bin/python"), "src/train.py", *overrides],
        cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return {"name": name, "process": process, "gpus": gpus, "log": log_path,
            "handle": handle, "started": time.time()}


def report(group):
    import wandb
    api = wandb.Api()
    runs = [r for r in api.runs("cath/voxelSmiles", filters={"group": group})]
    rows = []
    for run in runs:
        history = run.history(keys=["trainer/global_step", "val/likelihood_auc_znorm",
                                    "val/poc2mol/loss", "val/poc2mol/tanimoto",
                                    "val/zinc/loss"], pandas=False)
        auc = [h for h in history if h.get("val/likelihood_auc_znorm") is not None]
        loss = [h for h in history if h.get("val/poc2mol/loss") is not None]
        if not auc:
            continue
        best_auc = max(auc, key=lambda h: h["val/likelihood_auc_znorm"])
        best_loss = min(loss, key=lambda h: h["val/poc2mol/loss"]) if loss else {}
        rows.append({
            "name": run.name, "state": run.state,
            "best_auc_znorm": best_auc["val/likelihood_auc_znorm"],
            "auc_step": best_auc["trainer/global_step"],
            "best_poc2mol_loss": best_loss.get("val/poc2mol/loss"),
            "loss_step": best_loss.get("trainer/global_step"),
            "tanimoto_at_best_auc": best_auc.get("val/poc2mol/tanimoto"),
        })
    rows.sort(key=lambda r: -r["best_auc_znorm"])
    print(f"\n{'run':34s} {'AUC znorm':>10} {'@step':>7} {'poc2mol loss':>13} {'@step':>7} {'tanimoto':>9}")
    print("-" * 88)
    for r in rows:
        print(f"{r['name']:34s} {r['best_auc_znorm']:>10.4f} {r['auc_step']:>7} "
              f"{(r['best_poc2mol_loss'] or float('nan')):>13.4f} {(r['loss_step'] or -1):>7} "
              f"{(r['tanimoto_at_best_auc'] or float('nan')):>9.4f}")
    print(f"\nbaseline to beat: AUC znorm 0.7759 (A/B 9ch arm); composition-only readout 0.7615")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=6.0,
                        help="stop LAUNCHING new runs after this long (running ones finish)")
    parser.add_argument("--group", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if args.report:
        assert args.group, "--report needs --group"
        report(args.group)
        return

    group = args.group or f"sweep_s3_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir = os.path.join(REPO, "logs", "sweeps", group)
    os.makedirs(log_dir, exist_ok=True)
    configs = build_configs()
    print(f"group    : {group}")
    print(f"configs  : {len(configs)}")
    print(f"log dir  : {log_dir}")
    with open(os.path.join(log_dir, "configs.json"), "w") as h:
        json.dump(configs, h, indent=2)
    if args.dry_run:
        for c in configs:
            print("   ", c["name"], " ".join(c["overrides"]))
        return

    deadline = time.time() + args.hours * 3600
    queue, active, done = list(configs), [], []
    while queue or active:
        # reap
        for job in list(active):
            if job["process"].poll() is not None:
                job["handle"].close()
                elapsed = (time.time() - job["started"]) / 60
                status = "ok" if job["process"].returncode == 0 else f"FAIL({job['process'].returncode})"
                print(f"[{datetime.now():%H:%M:%S}] finished {job['name']:32s} {status} "
                      f"{elapsed:.1f} min", flush=True)
                active.remove(job)
                done.append(job["name"])
        # launch
        while queue and len(active) < len(GPU_PAIRS) and time.time() < deadline:
            busy = {g for job in active for g in job["gpus"]}
            free = next((p for p in GPU_PAIRS if not (set(p) & busy)), None)
            if free is None:
                break
            config = queue.pop(0)
            active.append(launch(config, free, group, log_dir))
            print(f"[{datetime.now():%H:%M:%S}] launched {config['name']:32s} on GPUs {free} "
                  f"({len(done)} done, {len(queue)} queued)", flush=True)
            time.sleep(20)  # stagger so four processes do not build indices simultaneously
        if time.time() >= deadline and queue:
            print(f"deadline reached; dropping {len(queue)} unstarted configs", flush=True)
            queue = []
        time.sleep(20)

    print(f"\nsweep complete: {len(done)} runs")
    try:
        report(group)
    except Exception as exc:
        print(f"report failed ({exc}); run with --report --group {group}")


if __name__ == "__main__":
    main()
