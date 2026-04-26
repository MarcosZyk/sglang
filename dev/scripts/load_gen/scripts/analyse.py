from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "prefill_time",
    "sum_decode_time",
    "num_local_cache_tokens",
    "num_global_cached_tokens",
}

OUTPUT_COLUMNS = [
    "agent_request_id",
    "prefill_total_time",
    "decode_total_time",
    "cache_hit_rate",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input_root = script_dir.parent / "result_128G"

    parser = argparse.ArgumentParser(
        description="Aggregate agent-level timing and cache statistics from result folders.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=default_input_root,
        help="Root directory containing per-method result folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory to write per-folder summary CSVs. Defaults to <input-root>/summary.",
    )
    return parser.parse_args()


def numeric_filename_sort_key(path: Path) -> tuple[int, int | str, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), path.name)
    return (1, stem, path.name)


def iter_result_folders(input_root: Path, output_root: Path) -> Iterable[Path]:
    for child in sorted(input_root.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        if child.resolve() == output_root.resolve():
            continue
        yield child


def validate_required_columns(fieldnames: list[str] | None, csv_path: Path) -> None:
    available = set(fieldnames or [])
    missing = sorted(REQUIRED_COLUMNS - available)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")


def aggregate_agent_csv(csv_path: Path) -> dict[str, float | str]:
    prefill_total_time = 0.0
    decode_total_time = 0.0
    num_local_cache_tokens = 0
    num_global_cached_tokens = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_required_columns(reader.fieldnames, csv_path)

        for row in reader:
            prefill_total_time += float(row["prefill_time"])
            decode_total_time += float(row["sum_decode_time"])
            num_local_cache_tokens += int(row["num_local_cache_tokens"])
            num_global_cached_tokens += int(row["num_global_cached_tokens"])

    cache_hit_rate = 0.0
    if num_global_cached_tokens > 0:
        cache_hit_rate = num_local_cache_tokens / num_global_cached_tokens

    return {
        "agent_request_id": csv_path.name,
        "prefill_total_time": prefill_total_time,
        "decode_total_time": decode_total_time,
        "cache_hit_rate": cache_hit_rate,
    }


def write_summary_csv(result_folder: Path, output_csv: Path) -> int:
    rows = [
        aggregate_agent_csv(csv_path)
        for csv_path in sorted(result_folder.glob("*.csv"), key=numeric_filename_sort_key)
    ]
    if not rows:
        return 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = (args.output_root or (input_root / "summary")).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"input root is not a directory: {input_root}")

    processed_any = False
    for result_folder in iter_result_folders(input_root, output_root):
        output_csv = output_root / f"{result_folder.name}.csv"
        row_count = write_summary_csv(result_folder, output_csv)
        if row_count == 0:
            print(f"skip {result_folder}: no agent csv files found")
            continue
        processed_any = True
        print(
            f"processed {result_folder.name}: {row_count} agent requests -> {output_csv}"
        )

    if not processed_any:
        print(f"no result folders with agent csv files found under {input_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
