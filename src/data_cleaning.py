"""Utilities for cleaning crash game round CSV files."""

from __future__ import annotations

import csv
import os
from typing import List


def load_sequence(path: str) -> List[float]:
    """Load multiplier column from a round CSV file."""
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "multiplier" in reader.fieldnames:
            return [float(row["multiplier"]) for row in reader]
        return [float(x) for row in reader for x in row]


def clean_sequence(seq: List[float]) -> List[float]:
    """Clean a multiplier sequence using simple heuristics."""
    if len(seq) < 2:
        return seq

    last = seq[-1]
    prev = seq[-2]

    if prev != 0:
        ratio = last / prev
    else:
        ratio = 0

    # Correct misplaced decimals (approx factor of 10)
    if 9 <= ratio <= 11:
        seq[-1] = last / 10
    elif 0 < ratio <= 0.11:
        seq[-1] = last * 10
    # Drop suspect final tick after a large spike
    elif last < 0.2 and prev > 2:
        seq = seq[:-1]
    elif last == 0:
        seq = seq[:-1]

    return seq


def clean_round_file(input_path: str, output_path: str) -> None:
    """Clean a single round CSV file and write the result."""
    seq = load_sequence(input_path)
    cleaned = clean_sequence(seq)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["multiplier"])
        for value in cleaned:
            writer.writerow([value])


def clean_directory(input_dir: str, output_dir: str) -> None:
    """Clean all round CSV files from one directory into another."""
    for root, _, files in os.walk(input_dir):
        for name in files:
            if not name.endswith(".csv"):
                continue
            in_path = os.path.join(root, name)
            rel = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, rel)
            clean_round_file(in_path, out_path)


if __name__ == "__main__":  # pragma: no cover - simple CLI
    import argparse

    parser = argparse.ArgumentParser(description="Clean crash game CSV data")
    parser.add_argument("input", help="Input directory with round CSV files")
    parser.add_argument("output", help="Directory to write cleaned files")
    args = parser.parse_args()

    clean_directory(args.input, args.output)
