#!/usr/bin/env python3
"""Aggregate per-complex summary.json into one wide features table.

Usage:
    python aggregate_results.py [--results DIR] [--out FILE]

Emits: features.parquet + features.tsv keyed by (target, complex_id).
Optionally joins MANIFEST.tsv (is_active, pchembl).
"""
import argparse, json, os, glob
import pandas as pd

DEFAULT_RESULTS = "/mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/gbsa-study/data/raw/complex_analyses"
DEFAULT_MANIFEST = "/mnt/netapp1/Store_othcxlwa/lwa_transfer/00_PRIORITY_discovery9_trajectories/MANIFEST.tsv"
DEFAULT_OUT = "/mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/gbsa-study/data/derived/features"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = []
    for target_dir in sorted(glob.glob(os.path.join(args.results, "*"))):
        target = os.path.basename(target_dir)
        for cid_dir in sorted(glob.glob(os.path.join(target_dir, "*"))):
            cid = os.path.basename(cid_dir)
            summary_path = os.path.join(cid_dir, "summary.json")
            if not os.path.exists(summary_path):
                continue
            with open(summary_path) as fh:
                s = json.load(fh)
            # flatten: keep only scalar-numeric or short list-len metadata
            row = {"target": target, "complex_id": cid}
            for k, v in s.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    row[k] = v
                elif isinstance(v, list):
                    row[f"{k}__n"] = len(v)
            rows.append(row)
    df = pd.DataFrame(rows)
    print(f"aggregated {len(df)} complexes across {df['target'].nunique() if len(df) else 0} targets")

    if os.path.exists(args.manifest):
        mani = pd.read_csv(args.manifest, sep="\t")
        mani = mani[["target","complex_id","is_active","pchembl","xtc_MB"]]
        df = df.merge(mani, on=["target","complex_id"], how="left")
        print(f"joined MANIFEST: {df['is_active'].notna().sum()} rows have is_active")

        # iter-3 FIX 8: MANIFEST.tsv has EMPTY is_active/pchembl for the 30 4A5S rows
        # (upstream ingestion issue). Recover them from gbsa-study/metadata.csv when
        # available, so downstream notebooks don't silently drop 4A5S.
        meta_p = "/mnt/lustre/scratch/nlsas/home/otras/hcx/lwa/gbsa-study/data/external/gbsa-study/data/raw/metadata.csv"
        if os.path.exists(meta_p):
            meta = pd.read_csv(meta_p)
            keep_cols = ["complex_id", "target", "is_active", "pchembl"]
            meta = meta[keep_cols].rename(
                columns={"is_active": "is_active_meta", "pchembl": "pchembl_meta"}
            )
            df = df.merge(meta, on=["complex_id", "target"], how="left")
            recov_before = int(df["is_active"].notna().sum())
            df["is_active"] = df["is_active"].astype("boolean").combine_first(
                df["is_active_meta"].astype("boolean")
            )
            df["pchembl"] = df["pchembl"].combine_first(df["pchembl_meta"])
            df = df.drop(columns=["is_active_meta", "pchembl_meta"], errors="ignore")
            recov_after = int(df["is_active"].notna().sum())
            if recov_after > recov_before:
                print(f"  iter-3 FIX 8: recovered is_active for {recov_after - recov_before} rows "
                      f"from metadata.csv (MANIFEST.tsv label gap; mostly 4A5S).")

    df.to_parquet(args.out + ".parquet", index=False)
    df.to_csv(args.out + ".tsv", sep="\t", index=False)
    print(f"wrote {args.out}.parquet and {args.out}.tsv  ({df.shape[0]} rows × {df.shape[1]} cols)")

if __name__ == "__main__":
    main()
