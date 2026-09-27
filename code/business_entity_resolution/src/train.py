#!/usr/bin/env python3
"""Train the matcher on dataset/train and save it to models/.

Usage:
    python3 src/train.py --train-dir dataset/train --model-dir models
"""
import argparse
from blocking import TOP_K_CANDIDATES
from pipeline import run_train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default="dataset/train")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--top-k", type=int, default=TOP_K_CANDIDATES,
                    help="Max candidates kept per S1 entity after blocking.")
    p.add_argument("--val-frac", type=float, default=0.2,
                    help="Fraction of S1 entities held out (by group) for threshold tuning.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default="cache",
                    help="Where to store/reuse preprocessed data (normalized sources, "
                         "candidates, features) across runs. Auto-invalidated per-file "
                         "if the underlying .tsv is newer than the cache. Pass "
                         "--no-cache to disable.")
    p.add_argument("--no-cache", action="store_true", help="Disable the on-disk cache entirely.")
    args = p.parse_args()
    run_train(args.train_dir, args.model_dir, top_k=args.top_k,
               val_frac=args.val_frac, seed=args.seed,
               cache_dir=None if args.no_cache else args.cache_dir)


if __name__ == "__main__":
    main()
