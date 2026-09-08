#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge sharded results.

Each shard writes ``metrics_shard<i>.csv`` and ``manifest_shard<i>.json`` into
the same experiment directory. This script concatenates them into the
unsharded filenames, checks that the shards agree on the configuration they
were run with, and reports any shard that is missing so a partial merge is
never mistaken for a complete one.

It deliberately does not regenerate figures. A shard holds a slice of the
cells, and a figure drawn from a partial slice is worse than no figure; rerun
the experiment unsharded, or add a per-experiment aggregation step, if a figure
is needed from a sharded run.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))


def merge_dir(d, expected_shards=None):
    merged = []
    for pattern in ("metrics", "sweep", "curves", "radius_history"):
        parts = sorted(glob.glob(os.path.join(d, f"{pattern}_shard*.csv")))
        if not parts:
            continue
        frames = []
        for p in parts:
            try:
                frames.append(pd.read_csv(p))
            except Exception as exc:
                print(f"  ! could not read {os.path.basename(p)}: {exc}")
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        dest = os.path.join(d, f"{pattern}.csv")
        df.to_csv(dest, index=False)
        merged.append((pattern, len(parts), len(df)))
        if expected_shards and len(parts) != expected_shards:
            print(f"  ! {pattern}: {len(parts)} shards found, "
                  f"{expected_shards} expected - the merge is INCOMPLETE")

    manifests = sorted(glob.glob(os.path.join(d, "manifest_shard*.json")))
    if manifests:
        blobs = []
        for p in manifests:
            try:
                with open(p) as fh:
                    blobs.append(json.load(fh))
            except Exception:
                pass
        if blobs:
            scales = {json.dumps(b.get("scale"), sort_keys=True) for b in blobs}
            if len(scales) > 1:
                print("  ! shards disagree on the scale configuration; "
                      "these results are not comparable")
            base = dict(blobs[0])
            base["shard"] = dict(merged_from=len(blobs))
            base["elapsed_s"] = sum(b.get("elapsed_s", 0) for b in blobs)
            with open(os.path.join(d, "manifest.json"), "w") as fh:
                json.dump(base, fh, indent=2, default=str)
    return merged


def main():
    scale = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SCALE", "smoke")
    expected = int(os.environ.get("SHARD_COUNT", "0")) or None

    dirs = sorted(glob.glob(os.path.join(RESULTS, f"EXP-*_{scale}")))
    if not dirs:
        print(f"no experiment directories for scale '{scale}' under {RESULTS}")
        return

    total = 0
    for d in dirs:
        merged = merge_dir(d, expected)
        if merged:
            name = os.path.basename(d)
            for pattern, n_parts, n_rows in merged:
                print(f"{name}: {pattern}.csv <- {n_parts} shards, {n_rows} rows")
            total += 1
    print(f"\nmerged {total} experiment directories")
    print("figures and summaries are not rebuilt from shards; "
          "run scripts/make_tables.py, or rerun unsharded for figures")


if __name__ == "__main__":
    main()
