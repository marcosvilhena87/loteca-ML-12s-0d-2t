"""Preprocess Loteca historical data for training."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from scripts.common import actual_symbol, ordered_top_symbols, read_semicolon_csv


HISTORY_PATH = Path("data/concursos_anteriores.csv")
OUTPUT_PATH = Path("output/processed_history.json")


def build_processed_history(rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_contest: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_contest[int(row["Concurso"])].append(row)

    contests_payload: List[Dict[str, object]] = []
    for contest_id in sorted(by_contest):
        games = by_contest[contest_id]
        game_payload: List[Dict[str, object]] = []

        per_top_ranked = {"top1": [], "top2": [], "top3": []}
        for game in games:
            actual = actual_symbol(game)
            top_map = ordered_top_symbols(game)
            record = {
                "Concurso": contest_id,
                "Jogo": int(game["Jogo"]),
                "Mandante": game["Mandante"],
                "Visitante": game["Visitante"],
                "actual": actual,
                "p_top": {
                    "top1": float(game["p(top1)"]),
                    "top2": float(game["p(top2)"]),
                    "top3": float(game["p(top3)"]),
                },
                "top_symbols": top_map,
                "hit": {
                    "top1": int(actual == top_map["top1"]),
                    "top2": int(actual == top_map["top2"]),
                    "top3": int(actual == top_map["top3"]),
                },
            }
            game_payload.append(record)
            for rank in ("top1", "top2", "top3"):
                per_top_ranked[rank].append(record)

        runs = {"top1": [], "top2": [], "top3": []}
        position_hits = {"top1": [], "top2": [], "top3": []}

        for rank in ("top1", "top2", "top3"):
            ordered_games = sorted(
                per_top_ranked[rank], key=lambda x: x["p_top"][rank], reverse=True
            )
            streak = 0
            for idx, g in enumerate(ordered_games, start=1):
                hit = int(g["hit"][rank])
                position_hits[rank].append({"position": idx, "hit": hit})
                if hit == 1:
                    streak += 1
                else:
                    if streak > 0:
                        runs[rank].append(streak)
                    streak = 0
            if streak > 0:
                runs[rank].append(streak)

        contests_payload.append(
            {
                "Concurso": contest_id,
                "games": game_payload,
                "position_hits": position_hits,
                "runs": runs,
            }
        )

    return {"contests": contests_payload, "num_contests": len(contests_payload)}


def main() -> None:
    rows = read_semicolon_csv(HISTORY_PATH)
    processed = build_processed_history(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(processed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[preprocess] concursos: {processed['num_contests']}")
    print(f"[preprocess] arquivo: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
