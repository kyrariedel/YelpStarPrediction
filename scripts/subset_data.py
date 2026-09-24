#!/usr/bin/env python3
"""Create subsets of the Yelp dataset for local tests."""

from pathlib import Path

# temporary hardocoded path; output directory relative to directory
# size of data = 1k, 10k, 100k, 1m; adjust later
SRC = Path(
    "/Users/kyrariedel/Desktop/2026/CS6240/Yelp JSON/"
    "yelp_dataset/yelp_academic_dataset_review.json"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "input"
SIZES = (1_000, 10_000, 100_000, 1_000_000)


def write_subset(n: int) -> Path:
    # write to output file based on size; 
    out = OUT_DIR / f"reviews_{n // 1000}k.json" if n < 1_000_000 else OUT_DIR / "reviews_1m.json"
    if n == 1_000:
        out = OUT_DIR / "reviews_1k.json"
    elif n == 10_000:
        out = OUT_DIR / "reviews_10k.json"
    elif n == 100_000:
        out = OUT_DIR / "reviews_100k.json"
    elif n == 1_000_000:
        out = OUT_DIR / "reviews_1m.json"
    else:
        out = OUT_DIR / f"reviews_{n}.json"

    # only write if DNE/meets requirements; check line count
    if out.exists() and out.stat().st_size > 0:
        with out.open() as f:
            existing = sum(1 for _ in f)
        if existing >= n:
            print(f"skip {out.name} (already has {existing} lines)")
            return out
        
    # write and breaking after hitting line count
    print(f"writing {out}: {n} lines")
    with SRC.open() as src, out.open("w") as dst:
        for i, line in enumerate(src):
            if i >= n:
                break
            dst.write(line)
    print(f"finished writing")
    return out

# error handle src file issues and create output directory if needed
def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing source reviews: {SRC}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n in SIZES:
        write_subset(n)


if __name__ == "__main__":
    main()