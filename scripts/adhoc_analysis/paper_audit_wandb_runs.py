import wandb, datetime
api = wandb.Api(timeout=90)

# What we're looking for, from the released checkpoints / configs:
#   poc2mol  : run dir 2025-04-21_18-13-26, ckpt epoch=173 global_step=13572, task_name 'poc2mol'
#   combined : task_name 'CombinedHiQBAggPropPoc2Mol', ckpt epoch=0 global_step=3,960,000
#              resumed from CombinedHiQBindCkptFrmPrevCombined/runs/2025-05-06_20-51-46
#   eval dir names embed run ids: zjhnye4j (May 11), i2eqbtyx (Mar 26)
TARGETS = {"zjhnye4j", "i2eqbtyx", "crz11hbc"}

for proj in ["poc2mol", "voxelSmiles"]:
    print("=" * 108)
    print(f"cath/{proj}")
    print("=" * 108)
    print(f"{'run id':10s} {'created (UTC)':20s} {'state':9s} {'steps':>10s} {'epoch':>6s}  {'task_name':32s} name")
    rows = []
    for r in api.runs(f"cath/{proj}"):
        s = r.summary
        step = s.get("_step") or s.get("trainer/global_step") or ""
        ep = s.get("epoch", "")
        tn = (r.config or {}).get("task_name", "")
        rows.append((r.created_at, r.id, r.state, step, ep, tn, r.name))
    for created, rid, state, step, ep, tn, name in sorted(rows):
        mark = "  <== " if rid in TARGETS else ""
        try: step = f"{int(step):,}"
        except Exception: step = str(step)[:10]
        try: ep = f"{int(ep)}"
        except Exception: ep = str(ep)[:6]
        print(f"{rid:10s} {str(created)[:19]:20s} {state:9s} {step:>10s} {ep:>6s}  {str(tn)[:32]:32s} {str(name)[:28]}{mark}")
    print()
