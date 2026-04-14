"""Generate constrained Loteca prediction and debugging logs."""

from __future__ import annotations

import itertools
import json
import csv
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from scripts.common import ordered_top_symbols, read_semicolon_csv, write_semicolon_csv

NEXT_CONTEST_PATH = Path("data/proximo_concurso.csv")
MODEL_PATH = Path("models/model.json")
PREDICTIONS_PATH = Path("output/predictions.csv")
DEBUG_PATH = Path("output/debug_report.txt")
EXPERIMENT_LOG_PATH = Path("output/experiment_log.csv")

HARD_COUNTS = {
    "triples": 2,
    "doubles": 0,
    "singles": 12,
    "top1_inclusions": 9,
    "top2_inclusions": 5,
    "top3_inclusions": 4,
}

SOFT_TARGETS = {"1": 8, "X": 5, "2": 5}
TRIPLE_POOL_SIZE = 7
STRUCTURAL_PENALTY_WEIGHT = 0.20
CONCENTRATION_PENALTY_WEIGHT = float(os.getenv("CONCENTRATION_PENALTY_WEIGHT", "0.12"))
STRUCTURAL_METRIC_WEIGHTS = {
    "top1": {
        "num_runs": 1.2,
        "mean_run_size": 0.8,
        "max_run": 1.0,
        "dist_1": 0.7,
        "dist_2": 0.8,
        "dist_3plus": 0.9,
    },
    "top2": {
        "num_runs": 0.9,
        "mean_run_size": 0.8,
        "max_run": 0.8,
        "dist_1": 1.3,
        "dist_2": 1.5,
        "dist_3plus": 0.5,
    },
    "top3": {
        "num_runs": 1.3,
        "mean_run_size": 0.7,
        "max_run": 0.8,
        "dist_1": 1.0,
        "dist_2": 1.3,
        "dist_3plus": 0.5,
    },
}
TRIPLE_CANDIDATE_WEIGHTS = {
    "uncertainty": 0.45,
    "proximity_top1_top2": 0.30,
    "structural_gain_estimate": 0.25,
}
ADAPTIVE_TRIPLE_POOL = {"low": 5, "high": 8, "uncertainty_threshold": 0.42, "min_uncertain_games": 6}


def build_game_features(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    games = []
    for row in sorted(rows, key=lambda r: int(r["Jogo"])):
        top_symbols = ordered_top_symbols(row)
        game = {
            "Concurso": int(row["Concurso"]),
            "Jogo": int(row["Jogo"]),
            "Mandante": row["Mandante"],
            "Visitante": row["Visitante"],
            "p_top": {
                "top1": float(row["p(top1)"]),
                "top2": float(row["p(top2)"]),
                "top3": float(row["p(top3)"]),
            },
            "top_symbol": top_symbols,
        }
        game["uncertainty"] = 1.0 - game["p_top"]["top1"]
        games.append(game)
    return games


def attach_positions(games: List[Dict[str, object]]) -> None:
    for rank in ("top1", "top2", "top3"):
        ordered = sorted(games, key=lambda g: g["p_top"][rank], reverse=True)
        for pos, game in enumerate(ordered, start=1):
            game.setdefault("positions", {})[rank] = pos


def score_game_rank(game: Dict[str, object], rank: str, model: Dict[str, object]) -> float:
    weights = model["weights"]
    pos = str(game["positions"][rank])
    pos_rate = model["position_rate"][rank].get(pos, model["overall_hit_rate"][rank])

    run_mean = model["run_stats"][rank]["mean"]
    run_norm = min(run_mean / 5.0, 1.0)

    return (
        weights["probability"] * game["p_top"][rank]
        + weights["position_rate"] * pos_rate
        + weights["run_boost"] * run_norm
    )


def evaluate_solution(
    games: List[Dict[str, object]],
    triples: Sequence[int],
    single_by_rank: Dict[str, Sequence[int]],
    model: Dict[str, object],
) -> Dict[str, object]:
    score = 0.0
    rank_score = {"top1": 0.0, "top2": 0.0, "top3": 0.0}
    symbol_counter: Counter[str] = Counter()
    simulated_hits = {"top1": [], "top2": [], "top3": []}

    for idx, game in enumerate(games):
        if idx in triples:
            for rank in ("top1", "top2", "top3"):
                rank_value = score_game_rank(game, rank, model)
                score += rank_value
                rank_score[rank] += rank_value
                symbol_counter[game["top_symbol"][rank]] += 1
                simulated_hits[rank].append(1)
        elif idx in single_by_rank["top1"]:
            rank_value = score_game_rank(game, "top1", model)
            score += rank_value
            rank_score["top1"] += rank_value
            symbol_counter[game["top_symbol"]["top1"]] += 1
            simulated_hits["top1"].append(1)
            simulated_hits["top2"].append(0)
            simulated_hits["top3"].append(0)
        elif idx in single_by_rank["top2"]:
            rank_value = score_game_rank(game, "top2", model)
            score += rank_value
            rank_score["top2"] += rank_value
            symbol_counter[game["top_symbol"]["top2"]] += 1
            simulated_hits["top1"].append(0)
            simulated_hits["top2"].append(1)
            simulated_hits["top3"].append(0)
        else:
            rank_value = score_game_rank(game, "top3", model)
            score += rank_value
            rank_score["top3"] += rank_value
            symbol_counter[game["top_symbol"]["top3"]] += 1
            simulated_hits["top1"].append(0)
            simulated_hits["top2"].append(0)
            simulated_hits["top3"].append(1)

    soft_penalty = sum(abs(symbol_counter[s] - SOFT_TARGETS[s]) for s in ("1", "X", "2")) * 0.05
    structural_details = structural_penalty_details(simulated_hits, model)
    return {
        "objective": score - soft_penalty - structural_details["penalty"],
        "score_bruto": score,
        "score_por_rank": rank_score,
        "soft_penalty": soft_penalty,
        "structural_penalty": structural_details["penalty"],
        "structural_penalty_components": structural_details["penalty_components"],
        "symbol_counter": dict(symbol_counter),
        "structural_profile": structural_details["profile_by_rank"],
        "structural_distance": structural_details["distance_by_rank"],
        "structural_targets": structural_details["targets_by_rank"],
    }


def run_lengths_from_hits(hits: Sequence[int]) -> List[int]:
    runs: List[int] = []
    streak = 0
    for hit in hits:
        if hit == 1:
            streak += 1
        elif streak > 0:
            runs.append(streak)
            streak = 0
    if streak > 0:
        runs.append(streak)
    return runs


def run_distribution(runs: Sequence[int]) -> Dict[str, float]:
    if not runs:
        return {"dist_1": 0.0, "dist_2": 0.0, "dist_3plus": 0.0}
    total = len(runs)
    return {
        "dist_1": sum(1 for run in runs if run == 1) / total,
        "dist_2": sum(1 for run in runs if run == 2) / total,
        "dist_3plus": sum(1 for run in runs if run >= 3) / total,
    }


def structural_profile_from_hits(hits: Sequence[int]) -> Dict[str, float]:
    runs = run_lengths_from_hits(hits)
    distribution = run_distribution(runs)
    return {
        "num_runs": float(len(runs)),
        "mean_run_size": float(sum(runs) / len(runs)) if runs else 0.0,
        "max_run": float(max(runs) if runs else 0),
        "dist_1": distribution["dist_1"],
        "dist_2": distribution["dist_2"],
        "dist_3plus": distribution["dist_3plus"],
    }


def structural_penalty_details(
    simulated_hits: Dict[str, Sequence[int]], model: Dict[str, object]
) -> Dict[str, object]:
    targets = model.get("structural_targets", {})
    metrics = ("num_runs", "mean_run_size", "max_run", "dist_1", "dist_2", "dist_3plus")
    distance_by_rank: Dict[str, Dict[str, float]] = {}
    profile_by_rank: Dict[str, Dict[str, float]] = {}
    total_weighted_distance = 0.0
    penalty_components = {"weighted_distance": 0.0, "concentration": 0.0}
    for rank in ("top1", "top2", "top3"):
        profile = structural_profile_from_hits(simulated_hits[rank])
        profile_by_rank[rank] = profile
        rank_targets = targets.get(rank, {})
        rank_metric_weights = STRUCTURAL_METRIC_WEIGHTS[rank]
        rank_distance = {}
        for metric in metrics:
            target = float(rank_targets.get(metric, 0.0))
            distance = abs(profile[metric] - target) * rank_metric_weights[metric]
            rank_distance[metric] = distance
            total_weighted_distance += distance
        distance_by_rank[rank] = rank_distance

    concentration_penalty = concentration_excess_penalty(simulated_hits["top2"]) + concentration_excess_penalty(
        simulated_hits["top3"]
    )
    penalty_components["weighted_distance"] = total_weighted_distance * STRUCTURAL_PENALTY_WEIGHT
    penalty_components["concentration"] = concentration_penalty * CONCENTRATION_PENALTY_WEIGHT
    targets_by_rank = {
        rank: {metric: float(targets.get(rank, {}).get(metric, 0.0)) for metric in metrics}
        for rank in ("top1", "top2", "top3")
    }
    return {
        "penalty": penalty_components["weighted_distance"] + penalty_components["concentration"],
        "profile_by_rank": profile_by_rank,
        "distance_by_rank": distance_by_rank,
        "targets_by_rank": targets_by_rank,
        "penalty_components": penalty_components,
    }


def concentration_excess_penalty(hits: Sequence[int]) -> float:
    runs = run_lengths_from_hits(hits)
    return sum(float((run - 1) ** 2) for run in runs if run >= 2)


def game_uncertainty_score(game: Dict[str, object]) -> float:
    p1 = game["p_top"]["top1"]
    p2 = game["p_top"]["top2"]
    p3 = game["p_top"]["top3"]
    return (p2 + p3) - 2 * (p1 - p3)


def estimate_structural_gain_for_triple(game: Dict[str, object]) -> float:
    p1 = game["p_top"]["top1"]
    p2 = game["p_top"]["top2"]
    p3 = game["p_top"]["top3"]
    return 0.6 * p2 + 0.4 * p3 - 0.5 * abs(p1 - p2)


def triple_candidate_score(game: Dict[str, object]) -> float:
    p1 = game["p_top"]["top1"]
    p2 = game["p_top"]["top2"]
    uncertainty = 1.0 - p1
    proximity_top1_top2 = 1.0 - (p1 - p2)
    structural_gain_estimate = estimate_structural_gain_for_triple(game)
    return (
        TRIPLE_CANDIDATE_WEIGHTS["uncertainty"] * uncertainty
        + TRIPLE_CANDIDATE_WEIGHTS["proximity_top1_top2"] * proximity_top1_top2
        + TRIPLE_CANDIDATE_WEIGHTS["structural_gain_estimate"] * structural_gain_estimate
    )


def adaptive_triple_pool_size(games: List[Dict[str, object]]) -> int:
    uncertain_games = sum(1 for game in games if game["uncertainty"] >= ADAPTIVE_TRIPLE_POOL["uncertainty_threshold"])
    if uncertain_games >= ADAPTIVE_TRIPLE_POOL["min_uncertain_games"]:
        return ADAPTIVE_TRIPLE_POOL["high"]
    return ADAPTIVE_TRIPLE_POOL["low"]


def optimize_ticket(games: List[Dict[str, object]], model: Dict[str, object]) -> Dict[str, object]:
    indices = list(range(len(games)))
    best = None

    singles_top1 = HARD_COUNTS["top1_inclusions"] - HARD_COUNTS["triples"]
    singles_top2 = HARD_COUNTS["top2_inclusions"] - HARD_COUNTS["triples"]
    singles_top3 = HARD_COUNTS["top3_inclusions"] - HARD_COUNTS["triples"]

    dynamic_pool_size = max(HARD_COUNTS["triples"], adaptive_triple_pool_size(games), TRIPLE_POOL_SIZE)
    ranked_candidates = sorted(indices, key=lambda i: triple_candidate_score(games[i]), reverse=True)
    triple_pool = ranked_candidates[:dynamic_pool_size]

    for triples in itertools.combinations(triple_pool, HARD_COUNTS["triples"]):
        remaining_after_triples = [i for i in indices if i not in triples]
        for top1_idxs in itertools.combinations(remaining_after_triples, singles_top1):
            rem_after_top1 = [i for i in remaining_after_triples if i not in top1_idxs]
            for top2_idxs in itertools.combinations(rem_after_top1, singles_top2):
                top3_idxs = tuple(i for i in rem_after_top1 if i not in top2_idxs)
                if len(top3_idxs) != singles_top3:
                    continue

                single_by_rank = {
                    "top1": set(top1_idxs),
                    "top2": set(top2_idxs),
                    "top3": set(top3_idxs),
                }
                evaluation = evaluate_solution(games, triples, single_by_rank, model)

                if best is None or evaluation["objective"] > best["objective"]:
                    best = {
                        "objective": evaluation["objective"],
                        "triples": set(triples),
                        "single_by_rank": single_by_rank,
                        "symbol_counter": evaluation["symbol_counter"],
                        "score_bruto": evaluation["score_bruto"],
                        "score_por_rank": evaluation["score_por_rank"],
                        "soft_penalty": evaluation["soft_penalty"],
                        "structural_penalty": evaluation["structural_penalty"],
                        "structural_penalty_components": evaluation["structural_penalty_components"],
                        "structural_profile": evaluation["structural_profile"],
                        "structural_distance": evaluation["structural_distance"],
                        "structural_targets": evaluation["structural_targets"],
                        "triple_pool_size": dynamic_pool_size,
                    }
    if best is None:
        raise RuntimeError("Não foi possível encontrar combinação válida.")
    return best


def format_pick(symbols: List[str]) -> str:
    ordered = [s for s in ("1", "X", "2") if s in symbols]
    if len(ordered) == 3:
        return "1X2"
    if len(ordered) == 2:
        return "".join(ordered)
    return ordered[0]


def build_outputs(games: List[Dict[str, object]], solution: Dict[str, object], model: Dict[str, object]) -> Tuple[List[Dict[str, object]], str]:
    rows: List[Dict[str, object]] = []
    debug_lines = []

    inc_counts = {"top1": 0, "top2": 0, "top3": 0}

    for idx, game in enumerate(games):
        included_ranks: List[str]
        if idx in solution["triples"]:
            included_ranks = ["top1", "top2", "top3"]
            tipo = "TRIPLO"
        elif idx in solution["single_by_rank"]["top1"]:
            included_ranks = ["top1"]
            tipo = "SECO"
        elif idx in solution["single_by_rank"]["top2"]:
            included_ranks = ["top2"]
            tipo = "SECO"
        else:
            included_ranks = ["top3"]
            tipo = "SECO"

        for rank in included_ranks:
            inc_counts[rank] += 1

        symbols = [game["top_symbol"][rank] for rank in included_ranks]
        palpite = format_pick(symbols)

        row = {
            "Concurso": game["Concurso"],
            "Jogo": game["Jogo"],
            "Mandante": game["Mandante"],
            "Visitante": game["Visitante"],
            "Palpite": palpite,
            "Tipo": tipo,
            "Ranks": "/".join(included_ranks),
            "Probabilidades": "|".join(f"{rank}:{game['p_top'][rank]:.4f}" for rank in included_ranks),
        }
        rows.append(row)

        debug_lines.append(
            f"Jogo {game['Jogo']:>2} | {game['Mandante']} x {game['Visitante']} | "
            f"ranks={included_ranks} | palpite={palpite} | "
            f"scores="
            + ", ".join(
                f"{rank}:{score_game_rank(game, rank, model):.4f}(pos={game['positions'][rank]})"
                for rank in ("top1", "top2", "top3")
            )
        )

    debug_summary = [
        "=== RESUMO DO PALPITE ===",
        f"Objective: {solution['objective']:.4f}",
        f"Score bruto: {solution['score_bruto']:.4f}",
        f"Score por rank: {solution['score_por_rank']}",
        f"Soft penalty: {solution['soft_penalty']:.4f}",
        f"Structural penalty: {solution['structural_penalty']:.4f}",
        f"Structural penalty components: {solution['structural_penalty_components']}",
        f"Contagem inclusões: {inc_counts}",
        f"Contagem símbolos (soft): {solution['symbol_counter']} alvo={SOFT_TARGETS}",
        "Perfil estrutural da aposta:",
        f"  top1={solution['structural_profile']['top1']}",
        f"  top2={solution['structural_profile']['top2']}",
        f"  top3={solution['structural_profile']['top3']}",
        "Structural targets usados na penalização:",
        f"  top1={solution['structural_targets']['top1']}",
        f"  top2={solution['structural_targets']['top2']}",
        f"  top3={solution['structural_targets']['top3']}",
        "Distância para structural_targets (abs por métrica):",
        f"  top1={solution['structural_distance']['top1']}",
        f"  top2={solution['structural_distance']['top2']}",
        f"  top3={solution['structural_distance']['top3']}",
        f"Hard constraints: {HARD_COUNTS}",
        f"Triple pool size usado: {solution['triple_pool_size']}",
        "",
        "=== DETALHE POR JOGO ===",
    ]
    debug_text = "\n".join(debug_summary + debug_lines)
    return rows, debug_text


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modelo não encontrado. Rode scripts/train_model.py primeiro.")

    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    next_rows = read_semicolon_csv(NEXT_CONTEST_PATH)

    games = build_game_features(next_rows)
    attach_positions(games)
    solution = optimize_ticket(games, model)
    predictions, debug_text = build_outputs(games, solution, model)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_semicolon_csv(
        PREDICTIONS_PATH,
        predictions,
        ["Concurso", "Jogo", "Mandante", "Visitante", "Palpite", "Tipo", "Ranks", "Probabilidades"],
    )
    DEBUG_PATH.write_text(debug_text, encoding="utf-8")
    append_experiment_log(solution)

    print(f"[predict] arquivo de palpites: {PREDICTIONS_PATH}")
    print(f"[predict] relatório debug: {DEBUG_PATH}")


def append_experiment_log(solution: Dict[str, object]) -> None:
    EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "timestamp",
        "weights",
        "structural_metric_weights",
        "structural_penalty_weight",
        "concentration_penalty_weight",
        "triple_pool_size",
        "objective",
        "score_bruto",
        "soft_penalty",
        "structural_penalty",
        "symbol_counts",
        "triple_games",
    ]
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "weights": json.dumps(TRIPLE_CANDIDATE_WEIGHTS, ensure_ascii=False),
        "structural_metric_weights": json.dumps(STRUCTURAL_METRIC_WEIGHTS, ensure_ascii=False),
        "structural_penalty_weight": STRUCTURAL_PENALTY_WEIGHT,
        "concentration_penalty_weight": CONCENTRATION_PENALTY_WEIGHT,
        "triple_pool_size": solution["triple_pool_size"],
        "objective": f"{solution['objective']:.6f}",
        "score_bruto": f"{solution['score_bruto']:.6f}",
        "soft_penalty": f"{solution['soft_penalty']:.6f}",
        "structural_penalty": f"{solution['structural_penalty']:.6f}",
        "symbol_counts": json.dumps(solution["symbol_counter"], ensure_ascii=False),
        "triple_games": ",".join(str(game_idx + 1) for game_idx in sorted(solution["triples"])),
    }

    existing_rows: List[Dict[str, str]] = []
    rewrite_file = False
    if EXPERIMENT_LOG_PATH.exists():
        with EXPERIMENT_LOG_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_headers = reader.fieldnames or []
            rewrite_file = existing_headers != headers
            if rewrite_file:
                for existing_row in reader:
                    existing_rows.append({header: existing_row.get(header, "") for header in headers})

    mode = "w" if rewrite_file or not EXPERIMENT_LOG_PATH.exists() else "a"
    with EXPERIMENT_LOG_PATH.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        if mode == "w":
            writer.writeheader()
            if existing_rows:
                writer.writerows(existing_rows)
        writer.writerow(row)


if __name__ == "__main__":
    main()
