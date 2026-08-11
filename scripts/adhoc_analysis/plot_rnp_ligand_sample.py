"""Contact sheet of a random sample of ligands from the Runs N' Poses eval set.

Purely for eyeballing what the filtered set actually contains — chemotype, size, how much of it
looks like a drug versus a leftover cofactor the CCD blocklist missed.

    ./venvPlixer/bin/python scripts/adhoc_analysis/plot_rnp_ligand_sample.py -n 200
"""

import argparse
import glob
import os

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")


def draw_options(legend_font: int, min_font: int):
    opts = rdMolDraw2D.MolDrawOptions()
    opts.legendFontSize = legend_font
    opts.minFontSize = min_font
    opts.maxFontSize = min_font + 12
    opts.bondLineWidth = 3
    opts.multipleBondOffset = 0.15
    # Tight padding and a small legend band: RDKit fits the drawing to the tile's aspect ratio,
    # so slack here is spent on whitespace rather than on the molecule.
    opts.padding = 0.02
    opts.legendFraction = 0.10
    opts.explicitMethyl = False
    return opts


def load_sample(parquet_dir: str, n: int, seed: int) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files in {parquet_dir}")
    cols = ["system_id", "smiles", "ligand_ccd_code", "novelty_tier", "max_tanimoto_to_train"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


def render(df: pd.DataFrame, path: str, per_row: int, size: tuple[int, int],
           legend_font: int, min_font: int) -> int:
    mols, legends = [], []
    for r in df.itertuples(index=False):
        mol = Chem.MolFromSmiles(r.smiles)
        if mol is None:
            continue
        rdDepictor.Compute2DCoords(mol)
        mols.append(mol)
        legends.append(f"{r.ligand_ccd_code}  T={r.max_tanimoto_to_train:.2f}")
    png = Draw.MolsToGridImage(
        mols, molsPerRow=per_row, subImgSize=size, legends=legends,
        drawOptions=draw_options(legend_font, min_font), useSVG=False, returnPNG=True,
    )
    data = png.data if hasattr(png, "data") else png
    with open(path, "wb") as fh:
        fh.write(data)
    return len(mols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default="../runs_n_poses/parquet/test")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="../runs_n_poses/ligand_sample.png")
    ap.add_argument("--per-row", type=int, default=5)
    ap.add_argument("--page-size", type=int, default=50,
                    help="molecules per output page; 200 in one image is not legible at fit-to-width")
    ap.add_argument("--width", type=int, default=520)
    ap.add_argument("--height", type=int, default=440)
    ap.add_argument("--legend-font", type=int, default=22)
    ap.add_argument("--min-font", type=int, default=18)
    args = ap.parse_args()

    df = load_sample(args.parquet_dir, args.n, args.seed)
    base, ext = os.path.splitext(args.out)
    size = (args.width, args.height)

    total, pages = 0, []
    for page, start in enumerate(range(0, len(df), args.page_size), start=1):
        chunk = df.iloc[start : start + args.page_size]
        path = f"{base}_p{page}{ext}" if len(df) > args.page_size else args.out
        n = render(chunk, path, args.per_row, size, args.legend_font, args.min_font)
        total += n
        pages.append((path, n))
        print(f"wrote {path}  ({n} molecules, {args.per_row} per row)")

    print(f"\nsample: {total} molecules over {len(pages)} page(s), "
          f"{df.ligand_ccd_code.nunique()} unique CCDs")
    print("novelty tiers:", df.novelty_tier.value_counts().sort_index().to_dict())
    # RDKit default palette, for reading the sheets: C black (unlabelled), N blue, O red,
    # S dark yellow, Cl green, F light green, Br brown, I purple, P orange.


if __name__ == "__main__":
    main()
