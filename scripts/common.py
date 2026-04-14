"""Common utilities for Loteca pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

NUMERIC_COLUMNS = {
    "Concurso",
    "Jogo",
    "1",
    "X",
    "2",
    "p(1)",
    "p(x)",
    "p(2)",
    "p(top1)",
    "p(top2)",
    "p(top3)",
    "top1",
    "top2",
    "top3",
}

SYMBOL_PRIORITY = {"1": 3, "2": 2, "X": 1}


def parse_decimal(value: str) -> float:
    value = (value or "").strip().replace(".", "").replace(",", ".")
    if value == "":
        return 0.0
    return float(value)


def format_decimal(value: float) -> str:
    return f"{value:.6f}".replace(".", ",")


def read_semicolon_csv(path: str | Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for raw in reader:
            row: Dict[str, object] = {}
            for key, value in raw.items():
                clean_key = key.strip()
                if clean_key in NUMERIC_COLUMNS:
                    if clean_key in {"Concurso", "Jogo", "1", "X", "2", "top1", "top2", "top3"}:
                        row[clean_key] = int(parse_decimal(value or "0"))
                    else:
                        row[clean_key] = parse_decimal(value or "0")
                else:
                    row[clean_key] = (value or "").strip()
            rows.append(row)
    return rows


def write_semicolon_csv(path: str | Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rank_symbols(row: Dict[str, object]) -> List[Tuple[str, float]]:
    probs = {
        "1": float(row["p(1)"]),
        "X": float(row["p(x)"]),
        "2": float(row["p(2)"]),
    }
    return sorted(probs.items(), key=lambda item: (item[1], SYMBOL_PRIORITY[item[0]]), reverse=True)


def ordered_top_symbols(row: Dict[str, object]) -> Dict[str, str]:
    ranked = rank_symbols(row)
    return {
        "top1": ranked[0][0],
        "top2": ranked[1][0],
        "top3": ranked[2][0],
    }


def actual_symbol(row: Dict[str, object]) -> str:
    if int(row.get("1", 0)) == 1:
        return "1"
    if int(row.get("2", 0)) == 1:
        return "2"
    return "X"
