"""Train heuristic model and persist aggregated metrics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

PROCESSED_PATH = Path("output/processed_history.json")
MODEL_PATH = Path("models/model.json")


def _safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def train_model(payload: Dict[str, object]) -> Dict[str, object]:
    contests: List[Dict[str, object]] = payload["contests"]

    position_stats = {"top1": defaultdict(list), "top2": defaultdict(list), "top3": defaultdict(list)}
    run_lengths = {"top1": [], "top2": [], "top3": []}
    overall_hit = {"top1": [], "top2": [], "top3": []}

    for contest in contests:
        for rank in ("top1", "top2", "top3"):
            for pos_hit in contest["position_hits"][rank]:
                position_stats[rank][pos_hit["position"]].append(pos_hit["hit"])
                overall_hit[rank].append(pos_hit["hit"])
            run_lengths[rank].extend(contest["runs"][rank])

    model = {
        "weights": {
            "probability": 0.70,
            "position_rate": 0.20,
            "run_boost": 0.10,
        },
        "position_rate": {
            rank: {str(pos): _safe_mean(vals) for pos, vals in sorted(position_stats[rank].items())}
            for rank in ("top1", "top2", "top3")
        },
        "run_stats": {
            rank: {
                "mean": _safe_mean(run_lengths[rank]),
                "max": max(run_lengths[rank]) if run_lengths[rank] else 0,
                "count": len(run_lengths[rank]),
            }
            for rank in ("top1", "top2", "top3")
        },
        "overall_hit_rate": {rank: _safe_mean(overall_hit[rank]) for rank in ("top1", "top2", "top3")},
        "metadata": {
            "num_contests": len(contests),
            "description": "Modelo heurístico com métricas de posição e runs.",
        },
    }
    return model


def main() -> None:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo {PROCESSED_PATH} não encontrado. Rode scripts/preprocess_data.py primeiro."
        )

    payload = json.loads(PROCESSED_PATH.read_text(encoding="utf-8"))
    model = train_model(payload)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[train] concursos usados: {model['metadata']['num_contests']}")
    print(f"[train] taxa média de acerto top1: {model['overall_hit_rate']['top1']:.4f}")
    print(f"[train] modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
