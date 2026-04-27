#!/usr/bin/env python3
"""Bees Algorithm runner with configurable parameters."""

import json
import argparse
from pathlib import Path
import numpy as np

from algorithms.bee.bee_algorithm import BeeAlgorithm
from planner.planner import Planner
from data.data_loader import DataLoader


def run_ba(
    num_exercises: int = 10,
    max_cycles: int = 100,
    population_size: int = 60,
    selected_sites: int = 12,
    elite_sites: int = 4,
    recruited_for_elite: int = 16,
    recruited_for_selected: int = 8,
    neighborhood_mutations: int = 1,
    neighborhood_radius: float = 0.35,
    intensity_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    muscle_group_weights: list[float] | tuple[float, ...] | None = None,
    random_seed: int = 42,
    output_dir: str = "algorithms/bee/results",
) -> Path:
    """Run BA and save history to JSON."""

    csv_path = Path("data/exrx_exercises_muscles_clean.csv")
    if not csv_path.exists():
        csv_path = Path("data/exrx_exercises_muscles_with_body_part.csv")

    loader = DataLoader(csv_path)
    exercises_array = loader.exercises()
    planner_intensity_weights = np.asarray(intensity_weights, dtype=np.float32).reshape(3, 1)
    planner = Planner(
        exercises_array,
        balance_weight=1.0,
        intensity_weights=planner_intensity_weights,
    )
    if muscle_group_weights is not None:
        parsed_muscle_group_weights = np.asarray(muscle_group_weights, dtype=np.float32).reshape(-1)
        if parsed_muscle_group_weights.size != planner.num_muscle_groups:
            raise ValueError(
                "muscle_group_weights must have exactly "
                f"{planner.num_muscle_groups} values (got {parsed_muscle_group_weights.size})."
            )
        planner.muscle_group_weights = parsed_muscle_group_weights.reshape(planner.num_muscle_groups, 1)

    ba = BeeAlgorithm(
        planner=planner,
        population_size=population_size,
        selected_sites=selected_sites,
        elite_sites=elite_sites,
        recruited_for_elite=recruited_for_elite,
        recruited_for_selected=recruited_for_selected,
        neighborhood_mutations=neighborhood_mutations,
        neighborhood_radius=neighborhood_radius,
        hamming_order_sensitive=True,
        random_seed=random_seed,
    )

    result = ba.run(num_exercises_to_plan=num_exercises, max_cycles=max_cycles)

    history_data = {
        "metadata": {
            "num_exercises": num_exercises,
            "num_cycles": len(result.best_cost_history),
            "total_fitness_evals": int(result.total_fitness_evals),
            "total_exercises_in_db": len(exercises_array),
            "muscle_groups": len(planner.idx2muscle_group),
            "ba_params": {
                "population_size": population_size,
                "selected_sites": selected_sites,
                "elite_sites": elite_sites,
                "recruited_for_elite": recruited_for_elite,
                "recruited_for_selected": recruited_for_selected,
                "neighborhood_mutations": neighborhood_mutations,
                "neighborhood_radius": neighborhood_radius,
                "intensity_weights": [float(v) for v in planner_intensity_weights.ravel()],
                "muscle_group_weights": [float(v) for v in planner.muscle_group_weights.ravel()],
                "random_seed": random_seed,
            },
        },
        "muscle_groups": planner.idx2muscle_group,
        "cycles": [],
    }

    for cycle_idx in range(len(result.best_cost_history)):
        population = result.population_history[cycle_idx]
        costs = result.cost_history[cycle_idx]
        best_cost = result.best_cost_history[cycle_idx]
        best_idx = int(np.argmin(costs))
        best_individual = population[best_idx]
        intensity_matrix = planner.get_intensity_matrix(best_individual)

        cycle_data = {
            "cycle": int(cycle_idx) + 1,
            "best_cost": float(best_cost),
            "fitness_evals": int(result.fitness_evals_history[cycle_idx]),
            "best_plan": [int(x) for x in best_individual],
            "exercise_names": [exercises_array[int(idx)].name for idx in best_individual],
            "exercise_urls": [exercises_array[int(idx)].url for idx in best_individual],
            "population_size": int(len(population)),
            "avg_cost": float(np.mean(costs)),
            "min_cost": float(np.min(costs)),
            "max_cost": float(np.max(costs)),
            "intensity_by_muscle": [
                {
                    "muscle": planner.idx2muscle_group[i],
                    "target_count": int(intensity_matrix[0, i]),
                    "synergist_count": int(intensity_matrix[1, i]),
                    "stabilizer_count": int(intensity_matrix[2, i]),
                    "total_intensity": int(intensity_matrix[0, i] + intensity_matrix[1, i] + intensity_matrix[2, i]),
                }
                for i in range(len(planner.idx2muscle_group))
                if intensity_matrix[0, i] > 0 or intensity_matrix[1, i] > 0 or intensity_matrix[2, i] > 0
            ],
        }
        history_data["cycles"].append(cycle_data)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history_file = output_path / "ba_full_history.json"
    with open(history_file, "w") as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)

    return history_file


def main():
    parser = argparse.ArgumentParser(description="Run Bees Algorithm with custom parameters")
    parser.add_argument("--num-exercises", type=int, default=10, help="Number of exercises in plan")
    parser.add_argument("--max-cycles", type=int, default=100, help="Maximum algorithm cycles")
    parser.add_argument("--population-size", type=int, default=60, help="Population size")
    parser.add_argument("--selected-sites", type=int, default=12, help="Selected sites")
    parser.add_argument("--elite-sites", type=int, default=4, help="Elite sites")
    parser.add_argument("--recruited-elite", type=int, default=16, help="Recruited for elite sites")
    parser.add_argument("--recruited-selected", type=int, default=8, help="Recruited for selected sites")
    parser.add_argument("--mutations", type=int, default=1, help="Neighborhood mutations")
    parser.add_argument("--radius", type=float, default=0.35, help="Neighborhood radius")
    parser.add_argument(
        "--intensity-weights",
        type=float,
        nargs=3,
        metavar=("TARGET", "SYNERGIST", "STABILIZER"),
        default=(1.0, 1.0, 1.0),
        help="Planner intensity weights for target/synergist/stabilizer",
    )
    parser.add_argument(
        "--muscle-group-weights",
        type=float,
        nargs="*",
        default=None,
        help="Optional full muscle-group weights vector (same length as inferred muscle groups)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="algorithms/bee/results", help="Output directory")

    args = parser.parse_args()

    output_file = run_ba(
        num_exercises=args.num_exercises,
        max_cycles=args.max_cycles,
        population_size=args.population_size,
        selected_sites=args.selected_sites,
        elite_sites=args.elite_sites,
        recruited_for_elite=args.recruited_elite,
        recruited_for_selected=args.recruited_selected,
        neighborhood_mutations=args.mutations,
        neighborhood_radius=args.radius,
        intensity_weights=tuple(args.intensity_weights),
        muscle_group_weights=args.muscle_group_weights,
        random_seed=args.seed,
        output_dir=args.output,
    )

    print(f"✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
