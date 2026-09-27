#!/usr/bin/env python3
"""Score dataset/test with a trained model and write output/*.tsv.

Usage:
    python3 src/predict.py --test-dir dataset/test --model-dir models --output-dir output
"""
import argparse
from pipeline import run_predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", default="dataset/test")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--top-k", type=int, default=None,
                    help="Candidate limit (defaults to the value saved with the model).")
    p.add_argument("--cache-dir", default="cache",
                    help="Where to store/reuse preprocessed data across runs (same "
                         "cache dir as train.py so the test-side normalization is "
                         "reused too if you re-predict). Pass --no-cache to disable.")
    p.add_argument("--no-cache", action="store_true", help="Disable the on-disk cache entirely.")
    args = p.parse_args()
    run_predict(args.test_dir, args.model_dir, args.output_dir, top_k=args.top_k,
                cache_dir=None if args.no_cache else args.cache_dir)


if __name__ == "__main__":
    main()
