from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data/metadata.csv")
    parser.add_argument("--output_csv", type=str, default="data/metadata_with_folds.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_col", type=str, default="grade")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    y = df[args.label_col].astype(int).values

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold = [-1] * len(df)
    for f, (_, val_idx) in enumerate(skf.split(df, y)):
        for i in val_idx:
            fold[i] = f

    df["fold"] = fold
    df.to_csv(args.output_csv, index=False)
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
