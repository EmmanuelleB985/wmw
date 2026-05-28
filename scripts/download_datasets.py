#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Download WMW evaluation datasets")
    parser.add_argument("--output-dir", type=str, default="data/eval")
    parser.add_argument("--n-synthetic", type=int, default=200)
    parser.add_argument("--max-per-source", type=int, default=200)
    parser.add_argument("--skip-external", action="store_true",
                        help="Only generate synthetic data")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    from wmw.datasets.prepare import prepare_all_datasets

    if args.skip_external:
        from wmw.datasets.prepare import prepare_synthetic
        from pathlib import Path
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        exs = prepare_synthetic(out, args.n_synthetic, args.seed)
        print(f"\nSynthetic only: {len(exs)} examples")
    else:
        all_sets = prepare_all_datasets(
            output_dir=args.output_dir,
            n_synthetic=args.n_synthetic,
            max_scienceqa=args.max_per_source,
            max_clevrer=args.max_per_source,
            max_mathvista=args.max_per_source,
            seed=args.seed,
            skip_download_errors=True,
        )
        print(f"\nDone. Total sources: {len(all_sets)}")


if __name__ == "__main__":
    main()
