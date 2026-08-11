import wandb, json
api = wandb.Api(timeout=90)

CAND = {
    "poc2mol": ["ljv96zyo", "zu8ks4hq", "2dlzm9os"],
    "voxelSmiles": ["zjhnye4j", "7yzj4c06", "55vlvc7x", "xkykyliz", "f4jjqi5m",
                    "ni6qdrd5", "ck2plqvg", "u9kthdvh", "t2gly4xp", "3cdi8t79"],
}

for proj, ids in CAND.items():
    for rid in ids:
        try:
            r = api.run(f"cath/{proj}/{rid}")
        except Exception as e:
            print(f"{rid}: ERROR {e}"); continue
        c = r.config or {}
        meta = {}
        try:
            for fn in r.files():
                if fn.name == "wandb-metadata.json":
                    meta = json.load(fn.download(replace=True, root="/tmp/wbmeta")); break
        except Exception:
            pass
        dcfg = c.get("data", {}) or {}
        def dig(d, *path):
            for p in path:
                if not isinstance(d, dict): return None
                d = d.get(p)
            return d
        train_path = (dig(dcfg, "train_dataset", "data_path")
                      or dig(dcfg, "train_dataset", "poc2mol_output_dataset", "complex_dataset", "data_path"))
        zinc = dig(dcfg, "train_dataset", "vox2smiles_dataset", "data_path")
        poc_ck = dig(dcfg, "train_dataset", "poc2mol_output_dataset", "ckpt_path")
        print(f"--- {proj}/{rid}  [{r.name}]  state={r.state}")
        print(f"      created   : {r.created_at}   runtime={r.summary.get('_runtime')}")
        print(f"      host      : {meta.get('host')}   started={meta.get('startedAt')}")
        print(f"      task_name : {c.get('task_name')}")
        print(f"      ckpt_path : {c.get('ckpt_path')}")
        if train_path: print(f"      train data: {train_path}")
        if zinc:       print(f"      zinc data : {zinc}")
        if poc_ck:     print(f"      poc2mol ck: {poc_ck}")
        args = (meta.get("args") or [])
        if args: print(f"      args      : {' '.join(args)[:150]}")
        print(f"      URL       : https://wandb.ai/cath/{proj}/runs/{rid}")
        print()
