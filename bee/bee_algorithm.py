from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from planner.planner import Planner


@dataclass(slots=True)
class BeeSearchResult:
	best_individual: np.ndarray
	best_cost: float
	population_history: list[np.ndarray]
	cost_history: list[np.ndarray]
	best_cost_history: list[float]

	def to_json_dict(self) -> dict[str, Any]:
		return {
			"best_individual": self.best_individual.astype(int).tolist(),
			"best_cost": float(self.best_cost),
			"population_history": [population.astype(int).tolist() for population in self.population_history],
			"cost_history": [costs.astype(float).tolist() for costs in self.cost_history],
			"best_cost_history": [float(value) for value in self.best_cost_history],
		}


class BeeAlgorithm:
	"""Bees Algorithm (BA) for searching exercise plans."""

	def __init__(
		self,
		planner: Planner,
		population_size: int = 60,
		selected_sites: int = 12,
		elite_sites: int = 4,
		recruited_for_elite: int = 16,
		recruited_for_selected: int = 8,
		neighborhood_mutations: int = 1,
		neighborhood_radius: float = 0.35,
		neighborhood_max_attempts: int = 25,
		hamming_order_sensitive: bool = True,
		random_seed: int | None = None,
	) -> None:
		if population_size <= 0:
			raise ValueError("population_size must be positive")
		if selected_sites <= 0:
			raise ValueError("selected_sites must be positive")
		if elite_sites < 0:
			raise ValueError("elite_sites must be non-negative")
		if elite_sites > selected_sites:
			raise ValueError("elite_sites cannot exceed selected_sites")
		if recruited_for_elite <= 0 or recruited_for_selected <= 0:
			raise ValueError("recruited bee counts must be positive")
		if neighborhood_mutations <= 0:
			raise ValueError("neighborhood_mutations must be positive")
		if neighborhood_radius <= 0:
			raise ValueError("neighborhood_radius must be positive")
		if neighborhood_max_attempts <= 0:
			raise ValueError("neighborhood_max_attempts must be positive")

		self.planner = planner
		self.population_size = population_size
		self.selected_sites = min(selected_sites, population_size)
		self.elite_sites = min(elite_sites, self.selected_sites)
		self.recruited_for_elite = recruited_for_elite
		self.recruited_for_selected = recruited_for_selected
		self.neighborhood_mutations = neighborhood_mutations
		self.neighborhood_radius = neighborhood_radius
		self.neighborhood_max_attempts = neighborhood_max_attempts
		self.hamming_order_sensitive = hamming_order_sensitive
		self._rng = np.random.default_rng(random_seed)

	def neighborhood_metric(self, individual_a: np.ndarray, individual_b: np.ndarray) -> float:
		"""Normalized Hamming distance between two plans in range [0, 1]."""
		if individual_a.shape != individual_b.shape:
			raise ValueError("Both individuals must have the same shape")
		if individual_a.size == 0:
			return 0.0

		if self.hamming_order_sensitive:
			return float(np.mean(individual_a != individual_b))

		# Order-insensitive variant: compare sorted exercise IDs (multiset comparison).
		sorted_a = np.sort(individual_a)
		sorted_b = np.sort(individual_b)
		return float(np.mean(sorted_a != sorted_b))

	def _generate_neighbor(self, individual: np.ndarray) -> np.ndarray:
		max_idx = self.planner.exercises.shape[0]

		for _ in range(self.neighborhood_max_attempts):
			neighbor = individual.copy()
			dim = neighbor.shape[0]
			mutation_count = min(self.neighborhood_mutations, dim)
			positions = self._rng.choice(dim, size=mutation_count, replace=False)

			for pos in positions:
				neighbor[pos] = int(self._rng.integers(0, max_idx))

			if self.neighborhood_metric(individual, neighbor) <= self.neighborhood_radius:
				return neighbor

		# Fallback: if radius is too strict for chosen mutation count, return last candidate.
		return neighbor

	def run(
		self,
		num_exercises_to_plan: int,
		max_cycles: int = 200,
		json_output_path: str | Path | None = None,
	) -> BeeSearchResult:
		if num_exercises_to_plan <= 0:
			raise ValueError("num_exercises_to_plan must be positive")
		if max_cycles <= 0:
			raise ValueError("max_cycles must be positive")

		population = self.planner.initialize_population(
			population_size=self.population_size,
			num_exercises_to_plan=num_exercises_to_plan,
		)
		costs = self.planner.evaluate_fitness(population).astype(np.float64)

		population_history: list[np.ndarray] = []
		cost_history: list[np.ndarray] = []
		best_cost_history: list[float] = []

		best_idx = int(np.argmin(costs))
		best_individual = population[best_idx].copy()
		best_cost = float(costs[best_idx])

		for _ in range(max_cycles):
			sorted_idx = np.argsort(costs)
			population = population[sorted_idx]
			costs = costs[sorted_idx]

			selected = population[: self.selected_sites]
			next_population: list[np.ndarray] = []

			# Local neighborhood search around selected sites.
			for site_rank, site in enumerate(selected):
				recruits = self.recruited_for_elite if site_rank < self.elite_sites else self.recruited_for_selected
				neighborhood: list[np.ndarray] = [site]
				for _ in range(recruits):
					neighborhood.append(self._generate_neighbor(site))

				neighborhood_arr = np.array(neighborhood, dtype=np.int32)
				neighborhood_costs = self.planner.evaluate_fitness(neighborhood_arr)
				best_local_idx = int(np.argmin(neighborhood_costs))
				next_population.append(neighborhood_arr[best_local_idx])

			remaining = self.population_size - len(next_population)
			if remaining > 0:
				scouts = self.planner.initialize_population(
					population_size=remaining,
					num_exercises_to_plan=num_exercises_to_plan,
				)
				next_population.extend(scouts)

			population = np.array(next_population, dtype=np.int32)
			costs = self.planner.evaluate_fitness(population).astype(np.float64)

			cycle_best_idx = int(np.argmin(costs))
			if float(costs[cycle_best_idx]) < best_cost:
				best_cost = float(costs[cycle_best_idx])
				best_individual = population[cycle_best_idx].copy()

			population_history.append(population.copy())
			cost_history.append(costs.copy())
			best_cost_history.append(best_cost)

		result = BeeSearchResult(
			best_individual=best_individual,
			best_cost=best_cost,
			population_history=population_history,
			cost_history=cost_history,
			best_cost_history=best_cost_history,
		)

		if json_output_path is not None:
			self.save_result_json(
				output_path=json_output_path,
				result=result,
				num_exercises_to_plan=num_exercises_to_plan,
				max_cycles=max_cycles,
			)

		return result

	def save_result_json(
		self,
		output_path: str | Path,
		result: BeeSearchResult,
		num_exercises_to_plan: int,
		max_cycles: int,
	) -> Path:
		output = Path(output_path).expanduser().resolve()
		output.parent.mkdir(parents=True, exist_ok=True)

		payload = {
			"algorithm": "ba",
			"params": {
				"n": self.population_size,
				"m": self.selected_sites,
				"e": self.elite_sites,
				"nep": self.recruited_for_elite,
				"nsp": self.recruited_for_selected,
				"neighborhood_mutations": self.neighborhood_mutations,
				"neighborhood_metric": "normalized_hamming",
				"hamming_order_sensitive": self.hamming_order_sensitive,
				"neighborhood_radius": self.neighborhood_radius,
				"neighborhood_max_attempts": self.neighborhood_max_attempts,
				"Cmax": max_cycles,
				"num_exercises_to_plan": num_exercises_to_plan,
			},
			"result": result.to_json_dict(),
		}

		output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
		return output
