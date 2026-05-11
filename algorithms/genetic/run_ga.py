#!/usr/bin/env python3
"""Genetic Algorithm runner with configurable parameters."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.genetic.genetic import Genetic
from algorithms.distribute_exercises.distribute_exercises_ilp import DistributeExercisesILP
from data.data_loader import DataLoader
from planner.planner import Planner
from utils.utils import create_logger


logger = create_logger(__name__)


def _build_intensity_by_muscle(planner: Planner, plan_indices: np.ndarray) -> list[dict[str, Any]]:
    intensity_matrix = planner.get_intensity_matrix(plan_indices)
    return [
        {
            "muscle": planner.idx2muscle_group[i],
            "target_count": int(intensity_matrix[0, i]),
            "synergist_count": int(intensity_matrix[1, i]),
            "stabilizer_count": int(intensity_matrix[2, i]),
            "total_intensity": int(intensity_matrix[0, i] + intensity_matrix[1, i] + intensity_matrix[2, i]),
        }
        for i in range(len(planner.idx2muscle_group))
        if intensity_matrix[0, i] > 0 or intensity_matrix[1, i] > 0 or intensity_matrix[2, i] > 0
    ]


def _build_daily_plans(
    planner: Planner,
    exercises_array: list,
    best_individual: np.ndarray,
    days: int,
    max_targets_per_day: int,
) -> list[dict[str, Any]]:
    distributor = DistributeExercisesILP(
        planner=planner,
        days=days,
        max_targets_per_day=max_targets_per_day,
    )
    daily_indices = distributor.split_exercises(best_individual.copy())
    daily_plans: list[dict[str, Any]] = []
    for day_idx, day_plan in enumerate(daily_indices, start=1):
        day_plan = np.asarray(day_plan, dtype=int)
        daily_plans.append(
            {
                "day": int(day_idx),
                "exercise_ids": [int(x) for x in day_plan],
                "exercise_names": [exercises_array[int(idx)].name for idx in day_plan],
                "exercise_urls": [exercises_array[int(idx)].url for idx in day_plan],
                "intensity_by_muscle": _build_intensity_by_muscle(planner, day_plan),
            }
        )
    return daily_plans


def run_ga(
    num_exercises: int = 10,
    days: int | None = None,
    exercises_per_day: int | None = None,
    max_targets_per_day: int = 5,
    max_cycles: int = 100,
    population_size: int = 60,
    elite_count: int = 4,
    crossover_type: str = "one-point",
    mixing_ratio: float = 0.5,
    mutation_chance: float = 0.1,
    intensity_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    muscle_group_weights: list[float] | tuple[float, ...] | None = None,
    normalize_helper_muscles: bool = True,
    random_seed: int = 42,
    output_dir: str = "algorithms/genetic/results",
) -> Path:
    """Run GA and save history in BA-compatible JSON schema."""
    logger.info(
        "Starting GA run: num_exercises=%s days=%s exercises_per_day=%s max_cycles=%s",
        num_exercises,
        days,
        exercises_per_day,
        max_cycles,
    )
    if (days is None) ^ (exercises_per_day is None):
        raise ValueError("days and exercises_per_day must be provided together.")
    if days is not None and exercises_per_day is not None:
        derived_exercises = int(days) * int(exercises_per_day)
        if num_exercises != derived_exercises:
            raise ValueError(
                "num_exercises must equal days * exercises_per_day "
                f"({num_exercises} != {derived_exercises})."
            )
    csv_path = Path("data/exrx_exercises_muscles_clean.csv")
    if not csv_path.exists():
        csv_path = Path("data/exrx_exercises_muscles_with_body_part.csv")

    loader = DataLoader(csv_path)
    exercises_array = loader.exercises()
    logger.info("Loaded exercises: count=%s source=%s", len(exercises_array), csv_path)
    planner_intensity_weights = np.asarray(intensity_weights, dtype=np.float32).reshape(3, 1)
    planner = Planner(
        exercises_array,
        balance_weight=1.0,
        intensity_weights=planner_intensity_weights,
        normalize_helper_muscles=normalize_helper_muscles,
    )
    if muscle_group_weights is not None:
        parsed_muscle_group_weights = np.asarray(muscle_group_weights, dtype=np.float32).reshape(-1)
        if parsed_muscle_group_weights.size != planner.num_muscle_groups:
            raise ValueError(
                "muscle_group_weights must have exactly "
                f"{planner.num_muscle_groups} values (got {parsed_muscle_group_weights.size})."
            )
        planner.muscle_group_weights = parsed_muscle_group_weights.reshape(planner.num_muscle_groups, 1)

    np.random.seed(random_seed)
    rng = np.random.default_rng(random_seed)

    ga = Genetic(
        fitness_function=planner.fitness_function,
        sequence_values=np.arange(len(exercises_array), dtype=np.int32),
        crossover_type=crossover_type,
        mixing_ratio=mixing_ratio,
        mutation_chance=mutation_chance,
        rng=rng,
    )

    planner.fitness_evaluations = 0
    population = planner.initialize_population(
        population_size=population_size,
        num_exercises_to_plan=num_exercises,
    )

    best_cost_so_far = float("inf")
    best_cost_history: list[float] = []
    population_history: list[np.ndarray] = []
    cost_history: list[np.ndarray] = []
    fitness_evals_history: list[int] = []

    for _ in range(max_cycles):
        population = ga.next_generation(population, elite_count=elite_count)
        costs = np.asarray(planner.evaluate_fitness(population), dtype=np.float64).reshape(population_size)

        cycle_best_idx = int(np.argmin(costs))
        cycle_best_cost = float(costs[cycle_best_idx])
        if cycle_best_cost < best_cost_so_far:
            best_cost_so_far = cycle_best_cost

        population_history.append(population.copy())
        cost_history.append(costs.copy())
        best_cost_history.append(best_cost_so_far)
        fitness_evals_history.append(int(planner.fitness_evaluations))

    history_data = {
        "metadata": {
            "days": int(days) if days is not None else None,
            "exercises_per_day": int(exercises_per_day) if exercises_per_day is not None else None,
            "num_cycles": max_cycles,
            "total_fitness_evals": int(planner.fitness_evaluations),
            "total_exercises_in_db": len(exercises_array),
            "muscle_groups": len(planner.idx2muscle_group),
            "ga_params": {
                "days": int(days) if days is not None else None,
                "exercises_per_day": int(exercises_per_day) if exercises_per_day is not None else None,
                "max_targets_per_day": int(max_targets_per_day),
                "population_size": population_size,
                "elite_count": elite_count,
                "crossover_type": crossover_type,
                "mixing_ratio": mixing_ratio,
                "mutation_chance": mutation_chance,
                "intensity_weights": [float(v) for v in planner_intensity_weights.ravel()],
                "muscle_group_weights": [float(v) for v in planner.muscle_group_weights.ravel()],
                "random_seed": random_seed,
            },
        },
        "muscle_groups": planner.idx2muscle_group,
        "cycles": [],
    }

    # Find best individual from final cycle
    final_costs = cost_history[-1]
    final_best_idx = int(np.argmin(final_costs))
    final_best_individual = population_history[-1][final_best_idx]

    for cycle_idx in range(max_cycles):
        cycle_population = population_history[cycle_idx]
        costs = cost_history[cycle_idx]
        best_idx = int(np.argmin(costs))
        best_individual = cycle_population[best_idx]
        intensity_by_muscle = _build_intensity_by_muscle(planner, best_individual)

        cycle_data = {
            "cycle": cycle_idx + 1,
            "best_cost": float(best_cost_history[cycle_idx]),
            "fitness_evals": int(fitness_evals_history[cycle_idx]),
            "best_plan": [int(x) for x in best_individual],
            "exercise_names": [exercises_array[int(idx)].name for idx in best_individual],
            "exercise_urls": [exercises_array[int(idx)].url for idx in best_individual],
            "population_size": int(len(cycle_population)),
            "population": cycle_population.astype(int).tolist(),
            "population_costs": [float(c) for c in costs],
            "avg_cost": float(np.mean(costs)),
            "min_cost": float(np.min(costs)),
            "max_cost": float(np.max(costs)),
            "intensity_by_muscle": intensity_by_muscle,
        }
        history_data["cycles"].append(cycle_data)

    # Perform exercise split once on final best individual
    if days is not None and exercises_per_day is not None:
        try:
            final_daily_plans = _build_daily_plans(
                planner,
                exercises_array,
                final_best_individual,
                days=int(days),
                max_targets_per_day=int(max_targets_per_day),
            )
            history_data["metadata"]["daily_plans"] = final_daily_plans
            logger.info("Final exercise split completed: %d days planned", len(final_daily_plans))
        except (RuntimeError, ValueError) as exc:
            history_data["metadata"]["daily_plans_error"] = str(exc)
            logger.warning("Final exercise split skipped: %s", str(exc))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history_file = output_path / "ga_full_history.json"
    with open(history_file, "w") as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)

    logger.info(
        "GA completed: cycles=%s best_cost=%.6f total_fitness_evals=%s",
        max_cycles,
        float(best_cost_history[-1]) if best_cost_history else float("nan"),
        int(planner.fitness_evaluations),
    )
    logger.info("GA history saved: %s", history_file)

    return history_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm with custom parameters")
    parser.add_argument("--num-exercises", type=int, default=10, help="Number of exercises in plan")
    parser.add_argument("--days", type=int, default=None, help="Number of training days")
    parser.add_argument("--exercises-per-day", type=int, default=None, help="Exercises per day")
    parser.add_argument(
        "--max-targets-per-day",
        type=int,
        default=3,
        help="Maximum distinct target muscles per day",
    )
    parser.add_argument("--max-cycles", type=int, default=100, help="Maximum algorithm cycles")
    parser.add_argument("--population-size", type=int, default=60, help="Population size")
    parser.add_argument("--elite-count", type=int, default=4, help="Elite individuals copied each generation")
    parser.add_argument(
        "--crossover-type",
        type=str,
        choices=["one-point", "two-point", "uniform"],
        default="one-point",
        help="Crossover operator",
    )
    parser.add_argument("--mixing-ratio", type=float, default=0.5, help="Uniform crossover mixing ratio")
    parser.add_argument("--mutation-chance", type=float, default=0.1, help="Mutation probability per gene")
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
    parser.add_argument(
        "--normalize-helper-muscles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize synergist and stabilizer contributions across muscles in each role",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="algorithms/genetic/results", help="Output directory")

    args = parser.parse_args()

    output_file = run_ga(
        num_exercises=args.num_exercises,
        days=args.days,
        exercises_per_day=args.exercises_per_day,
        max_targets_per_day=args.max_targets_per_day,
        max_cycles=args.max_cycles,
        population_size=args.population_size,
        elite_count=args.elite_count,
        crossover_type=args.crossover_type,
        mixing_ratio=args.mixing_ratio,
        mutation_chance=args.mutation_chance,
        intensity_weights=tuple(args.intensity_weights),
        muscle_group_weights=args.muscle_group_weights,
        normalize_helper_muscles=args.normalize_helper_muscles,
        random_seed=args.seed,
        output_dir=args.output,
    )

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
