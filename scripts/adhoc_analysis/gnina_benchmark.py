"""Cross-dock the PLINDER panel with Gnina and score the ranking.

Gnina is Vina's docking engine plus a CNN rescorer, so running it on the **same receptors, the
same ligand conformers and the same boxes as `vina_benchmark.py`** isolates the scoring function
as the only difference between the two. This script therefore reuses `../vina_bench/plinder`
rather than preparing its own inputs -- do not regenerate them, or the comparison stops being
controlled.

THREE SCORES ARE RECORDED, because they are not interchangeable:
  affinity      Vina-like empirical score, kcal/mol, LOWER is better  -> negated on collect
  CNNscore      CNN pose quality, 0-1, higher better. Says "is this pose right", not "does it
                bind", so it is the weakest ranker of the three for virtual screening.
  CNNaffinity   CNN predicted pK, higher better. This is what Gnina's documentation recommends
                for ranking, and it is the primary readout here.

⚠️ Gnina prints "initial pose not within box" for most cross-docked pairs. That is expected and
harmless: the input conformer is embedded at the origin-ish and the sampler randomises into the
box. It is not a failure and must not be filtered out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

TOOLS = "/home/judewells/plixer_outer/tools"
# The "static" release is not static: it needs cuDNN 9 and the CUDA 12 runtime, which are
# installed under tools/cudnn9 and must NOT be put on venvPlixer's path -- torch 2.3.1 needs
# cuDNN 8 and breaks if it resolves 9 instead.
LIBS = ":".join(sorted(
    os.path.join(TOOLS, "cudnn9", "nvidia", package, "lib")
    for package in os.listdir(os.path.join(TOOLS, "cudnn9", "nvidia"))
    if os.path.isdir(os.path.join(TOOLS, "cudnn9", "nvidia", package, "lib"))))

ROW = re.compile(r"^\s*1\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)")


def dock(job):
    system_id, candidate, receptor, ligand, centre, size, exhaustiveness, gpu, out, seed = job
    if os.path.exists(out):
        return None
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = LIBS + ":" + env.get("LD_LIBRARY_PATH", "")
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    command = [os.path.join(TOOLS, "gnina"),
               "-r", receptor, "-l", ligand,
               "--center_x", f"{centre[0]}", "--center_y", f"{centre[1]}",
               "--center_z", f"{centre[2]}",
               "--size_x", f"{size[0]}", "--size_y", f"{size[1]}", "--size_z", f"{size[2]}",
               "--exhaustiveness", str(exhaustiveness), "--num_modes", "1",
               "--seed", str(seed)]
    record = {"system_id": system_id, "candidate": candidate}
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=600, env=env)
        for line in done.stdout.splitlines():
            hit = ROW.match(line)
            if hit:
                record.update(affinity=float(hit.group(1)), intramol=float(hit.group(2)),
                              cnn_score=float(hit.group(3)), cnn_affinity=float(hit.group(4)))
                break
        else:
            record["error"] = (done.stderr or done.stdout)[-300:]
    except subprocess.TimeoutExpired:
        record["error"] = "timeout"
    except Exception as error:                                   # noqa: BLE001
        record["error"] = str(error)
    with open(out, "w") as handle:
        json.dump(record, handle)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vina_work", default="../vina_bench/plinder",
                        help="reuse its receptors/ligands/boxes so the comparison is controlled")
    parser.add_argument("--work", default="../gnina_bench/plinder")
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--per_gpu", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="pilot on the first N pairs")
    args = parser.parse_args()

    meta = json.load(open(f"{args.vina_work}/manifest.json"))
    boxes = json.load(open(f"{args.vina_work}/boxes.json"))
    failed = set(meta["failed_ligands"])
    os.makedirs(f"{args.work}/scores", exist_ok=True)

    jobs = []
    for system_id in meta["system_ids"]:
        if system_id not in boxes:
            continue
        centre, size = boxes[system_id]
        for candidate in meta["panel_ids"]:
            if candidate in failed:
                continue
            out = f"{args.work}/scores/{system_id}__{candidate}.json"
            if os.path.exists(out):
                continue
            gpu = len(jobs) % args.gpus
            jobs.append((system_id, candidate,
                         f"{args.vina_work}/receptors/{system_id}.pdbqt",
                         f"{args.vina_work}/ligands/{candidate}.pdbqt",
                         centre, size, args.exhaustiveness, gpu, out, args.seed))
    if args.limit:
        jobs = jobs[:args.limit]

    workers = args.gpus * args.per_gpu
    print(f"docking {len(jobs)} pairs on {args.gpus} GPUs x {args.per_gpu} "
          f"(exhaustiveness {args.exhaustiveness})", flush=True)
    done, failures = 0, 0
    # Threads, not processes: each worker only waits on a gnina subprocess, so the GIL is
    # irrelevant and threads keep the per-GPU dispatch trivially shared.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(dock, job) for job in jobs]
        for future in as_completed(futures):
            record = future.result()
            done += 1
            if record and "error" in record:
                failures += 1
            if done % 250 == 0:
                print(f"  {done}/{len(jobs)}  ({failures} failed)", flush=True)
    print(f"docked {done} pairs, {failures} failed")


if __name__ == "__main__":
    sys.exit(main())
