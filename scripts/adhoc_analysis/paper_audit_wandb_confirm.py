import wandb, re, os
api = wandb.Api(timeout=90)

# 1) trace remaining ancestors
print("=== ancestor ckpt_paths ===")
for proj, rid in [("voxelSmiles", "crz11hbc"), ("voxelSmiles", "s7xnxqhu"),
                  ("voxelSmiles", "lryunro2"), ("voxelSmiles", "q86jzq81")]:
    try:
        r = api.run(f"cath/{proj}/{rid}")
        c = r.config or {}
        print(f"  {rid} [{r.name}] task={c.get('task_name')} created={str(r.created_at)[:19]}")
        print(f"        ckpt_path: {c.get('ckpt_path')}")
    except Exception as e:
        print(f"  {rid}: ERROR {e}")

# 2) confirm which wandb run wrote which hydra output dir, from console logs
print("\n=== hydra 'Output dir' recorded in each run's console log ===")
WANT = [("poc2mol", "ljv96zyo"), ("poc2mol", "zu8ks4hq"),
        ("voxelSmiles", "zjhnye4j"), ("voxelSmiles", "55vlvc7x"), ("voxelSmiles", "crz11hbc")]
pat = re.compile(r"(logs/[A-Za-z0-9_]+/runs/[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}[A-Za-z_]*)")
for proj, rid in WANT:
    found = set()
    try:
        r = api.run(f"cath/{proj}/{rid}")
        for fn in r.files():
            if fn.name.endswith("output.log"):
                p = fn.download(replace=True, root=f"/tmp/wblog/{rid}")
                with open(p.name, errors="ignore") as fh:
                    for line in fh:
                        for m in pat.findall(line):
                            found.add(m)
                break
    except Exception as e:
        print(f"  {rid}: ERROR {type(e).__name__}: {e}"); continue
    print(f"  {proj}/{rid}: {sorted(found) if found else 'NO output.log / no match'}")
