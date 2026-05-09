"""
ILP-based exercise distributor
===============================
Distributes a pre-selected set of n = k·p exercises across k training days
(p exercises each) using Mixed-Integer Linear Programming.

Each day is constrained to at most ``max_targets_per_day`` distinct *target*
muscle groups, ensuring each session stays focused.

Objective (minimised)
---------------------
    Σ_d  [ y · Σ_g |L_d[g] - mean(L_d)|  -  L_d · alpha ]

where
    L_d[g]   = total beta-weighted load on muscle g on day d
    alpha    = planner.muscle_group_weights  (G,)   preference vector
    beta        = planner.intensity_weights     (3,)   role weights
    y        = planner.balance_weight               balance coefficient

The L1 norm is an exact linearisation of the spread term - it keeps the
problem a pure MILP solvable by the HiGHS solver bundled with cvxpy.

Dependency
----------
    cvxpy is not listed in pyproject.toml yet; add it:

        uv add cvxpy        # or:  pip install cvxpy
"""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np

from planner.planner import Planner


class DistributeExercisesILP:
    """
    Distribute exercise indices across training days via MILP.

    The interface mirrors ``DistributeExercisesMetaheuristic``:
    ``split_exercises`` receives an array of exercise indices and returns a
    list of per-day index arrays.

    Parameters
    ----------
    planner : Planner
        Fully initialised Planner instance.  Provides the exercise pool,
        muscle-group index, and all cost-function parameters (alpha, beta, y).
    days : int
        Number of training days.
    max_targets_per_day : int
        Maximum distinct *target* muscle groups allowed per day. Default: 3.
    """

    def __init__(
        self,
        planner: Planner,
        days: int,
        max_targets_per_day: int = 3,
    ) -> None:
        self.planner = planner
        self.days = days
        self.max_targets_per_day = max_targets_per_day

    def _build_load_matrix(self, exercise_indices: np.ndarray) -> np.ndarray:
        """
        Compute the beta-weighted load matrix for the selected exercises.

        Returns
        -------
        load : np.ndarray, shape (n, G)
            load[i, g] = weighted contribution of exercise i to muscle g.
        """
        n = len(exercise_indices)
        G = self.planner.num_muscle_groups
        beta = self.planner.intensity_weights.ravel()   # (3,)
        load = np.zeros((n, G), dtype=float)

        for i, ex_idx in enumerate(exercise_indices):
            exercise = self.planner.exercises[ex_idx]
            for muscle in exercise.targets:
                j = self.planner.muscle_group2idx[
                    self.planner._normalize_muscle_name(muscle)
                ]
                load[i, j] += beta[0]
            for muscle in exercise.synergists:
                j = self.planner.muscle_group2idx[
                    self.planner._normalize_muscle_name(muscle)
                ]
                load[i, j] += beta[1]
            for muscle in exercise.stabilizers:
                j = self.planner.muscle_group2idx[
                    self.planner._normalize_muscle_name(muscle)
                ]
                load[i, j] += beta[2]

        return load

    def _build_target_indicator(self, exercise_indices: np.ndarray) -> np.ndarray:
        """
        Build a binary indicator matrix for target muscle groups.

        Returns
        -------
        target_ind : np.ndarray, shape (n, TG)
            target_ind[i, t] = 1 iff target group t appears in exercise i's
            targets list.
        """
        target_groups = sorted({
            self.planner._normalize_muscle_name(g)
            for idx in exercise_indices
            for g in self.planner.exercises[idx].targets
        })
        tg_idx = {g: t for t, g in enumerate(target_groups)}

        n   = len(exercise_indices)
        TG  = len(target_groups)
        target_ind = np.zeros((n, TG), dtype=float)

        for i, ex_idx in enumerate(exercise_indices):
            for muscle in self.planner.exercises[ex_idx].targets:
                key = self.planner._normalize_muscle_name(muscle)
                target_ind[i, tg_idx[key]] = 1.0

        return target_ind

    def split_exercises(self, exercise_indices: np.ndarray) -> list[np.ndarray]:
        """
        Distribute exercise indices across days using MILP.

        Parameters
        ----------
        exercise_indices : np.ndarray of int, shape (n,)
            Indices into ``self.planner.exercises`` for the exercises to
            distribute.  ``n`` must be divisible by ``self.days``.

        Returns
        -------
        list[np.ndarray]
            List of ``self.days`` arrays, each containing the exercise indices
            assigned to that day (order within each day is arbitrary).

        Raises
        ------
        ValueError
            If ``n`` is not divisible by ``self.days`` or if the ILP is
            infeasible (e.g. ``max_targets_per_day`` is too restrictive).
        RuntimeError
            If no supported MILP solver is available.
        """
        n = len(exercise_indices)
        if n % self.days != 0:
            raise ValueError(
                f"Number of exercises ({n}) must be divisible by "
                f"days ({self.days})."
            )
        p = n // self.days

        # Cost-function parameters from the Planner
        G     = self.planner.num_muscle_groups
        alpha = self.planner.muscle_group_weights.ravel()   # (G,)
        gamma = float(self.planner.balance_weight)

        # Precomputed constant matrices
        load       = self._build_load_matrix(exercise_indices)       # (n, G)
        target_ind = self._build_target_indicator(exercise_indices)  # (n, TG)
        TG         = target_ind.shape[1]

        # Adjusted load: removes the per-exercise mean contribution so that
        # the deviation  L_d[g] - mean(L_d)  stays linear in z (no auxiliary
        # mean variable needed).
        adj_load = load - load.mean(axis=1, keepdims=True)           # (n, G)

        # z[i, d]  ∈ {0, 1}  - exercise i is assigned to day d
        # y[t, d]  ∈ {0, 1}  - target group t is active on day d
        # w[d, g]  ≥  0      - upper-bounds |L_d[g] - mean(L_d)|  (L1 abs)
        z = cp.Variable((n,  self.days), boolean=True, name="z")
        y = cp.Variable((TG, self.days), boolean=True, name="y")
        w = cp.Variable((self.days, G),  nonneg=True,  name="w")

        dev = z.T @ adj_load    # (days, G) - affine in z

        constraints = [
            # C1: every exercise is assigned to exactly one day
            cp.sum(z, axis=1) == 1,
            # C2: every day has exactly p exercises
            cp.sum(z, axis=0) == p,
            # C3: at most max_targets_per_day distinct target groups per day
            cp.sum(y, axis=0) <= self.max_targets_per_day,
            # C4: if any exercise with target t is on day d, y[t, d] must be 1
            #     LHS ≤ p guarantees that a single active exercise forces y = 1
            target_ind.T @ z <= p * y,
            # C5: linearise |dev[d, g]| via the standard abs-value trick
            w >=  dev,
            w >= -dev,
        ]

        # Per-day load matrix L[d, g] = Σ_i z[i,d] · load[i,g]
        L = z.T @ load      # (days, G)

        objective = cp.Minimize(
            gamma * cp.sum(w)       # L1 variance across muscle groups per day
            - cp.sum(L @ alpha)     # preference: reward high-alpha muscle load
        )

        problem = cp.Problem(objective, constraints)

        # Solver priority: HiGHS ships with cvxpy ≥ 1.4 and handles MILP well
        _SOLVER_PRIORITY = [cp.HIGHS, cp.SCIP, cp.CPLEX, cp.GUROBI, cp.ECOS_BB]
        for solver in _SOLVER_PRIORITY:
            if solver not in cp.installed_solvers():
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    problem.solve(solver=solver, verbose=False)
                if problem.status in ("optimal", "optimal_inaccurate"):
                    break
            except cp.error.SolverError:
                continue
        else:
            raise RuntimeError(
                "No suitable MILP solver is available. "
                "Ensure cvxpy ≥ 1.4 is installed:  uv add cvxpy"
            )

        if z.value is None:
            raise ValueError(
                f"ILP returned no solution (status: {problem.status}). "
                "The problem may be infeasible - try increasing "
                "max_targets_per_day or reducing the number of days."
            )

        # Map the (n, days) assignment matrix back to per-day index lists
        assignment = np.argmax(np.round(z.value).astype(int), axis=1)  # (n,)
        return [exercise_indices[assignment == d] for d in range(self.days)]
