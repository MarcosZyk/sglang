#!/usr/bin/env python3
"""Flatten aggregated output_aone JSON files into per-call records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Flatten aggregated output_aone JSON files into per-call records, "
            "sort by end_time, and recompute wait_time."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=base_dir / "output_aone",
        help="Directory containing aggregated JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "output_aone_flattened",
        help="Directory where flattened JSON files will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern used to select input JSON files.",
    )
    return parser.parse_args()


def load_blocks(file_path: Path) -> List[Dict[str, Any]]:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{file_path}: invalid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(f"{file_path}: top-level JSON value must be a list")

    for block_idx, block in enumerate(data):
        if not isinstance(block, dict):
            raise ValueError(
                f"{file_path}: block {block_idx} must be an object, got {type(block).__name__}"
            )
    return data


def validate_block_n(file_path: Path, block_idx: int, block: Dict[str, Any]) -> int:
    n = block.get("n")
    if not isinstance(n, int) or n <= 0:
        raise ValueError(
            f"{file_path}: block {block_idx} has invalid n={n!r}; expected a positive integer"
        )
    return n


def scatter_block(
    file_path: Path,
    block_idx: int,
    block: Dict[str, Any],
    total_n: int,
) -> Iterable[Dict[str, Any]]:
    n = validate_block_n(file_path, block_idx, block)
    per_call_fields: Dict[str, List[Any]] = {}
    scalar_fields: Dict[str, Any] = {}

    for key, value in block.items():
        if key == "n":
            continue
        if isinstance(value, list):
            if len(value) != n:
                raise ValueError(
                    f"{file_path}: block {block_idx} field {key!r} length={len(value)} "
                    f"does not match n={n}"
                )
            per_call_fields[key] = value
        else:
            scalar_fields[key] = value

    for required_key in ("start_time", "duration"):
        if required_key not in per_call_fields:
            raise ValueError(
                f"{file_path}: block {block_idx} is missing required array field {required_key!r}"
            )

    for call_idx in range(n):
        record: Dict[str, Any] = {"n": total_n}
        record.update(scalar_fields)

        for key, values in per_call_fields.items():
            record[key] = values[call_idx]

        try:
            start_time = float(record["start_time"])
            duration = float(record["duration"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{file_path}: block {block_idx} call {call_idx} has non-numeric "
                "start_time or duration"
            ) from exc

        record["end_time"] = start_time + duration
        yield record


def recompute_wait_times(records: List[Dict[str, Any]]) -> None:
    for idx, record in enumerate(records):
        if idx + 1 >= len(records):
            record["wait_time"] = 0.0
            continue

        next_start = float(records[idx + 1]["start_time"])
        current_end = float(record["end_time"])
        record["wait_time"] = max(0.0, next_start - current_end)


def flatten_file(file_path: Path) -> List[Dict[str, Any]]:
    blocks = load_blocks(file_path)
    total_n = sum(validate_block_n(file_path, idx, block) for idx, block in enumerate(blocks))
    flattened: List[Dict[str, Any]] = []

    for block_idx, block in enumerate(blocks):
        flattened.extend(scatter_block(file_path, block_idx, block, total_n))

    flattened.sort(key=lambda item: item["end_time"])
    recompute_wait_times(flattened)
    return flattened


def process_directory(input_dir: Path, output_dir: Path, pattern: str) -> int:
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    for file_path in sorted(input_dir.glob(pattern)):
        if file_path.suffix.lower() != ".json":
            continue

        flattened = flatten_file(file_path)
        output_path = output_dir / file_path.name
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(flattened, f, ensure_ascii=False, indent=2)
            f.write("\n")
        processed += 1

    return processed


def main() -> int:
    args = parse_args()
    processed = process_directory(args.input_dir, args.output_dir, args.pattern)
    print(
        f"Processed {processed} file(s) from {args.input_dir} into {args.output_dir}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
