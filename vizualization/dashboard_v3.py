#!/usr/bin/env python3
"""Interactive dashboard - Bees Algorithm + MuscleMap visualization."""

import json
import sys
from pathlib import Path
from collections import Counter
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))
from algorithms.bee.run_ba import run_ba
from algorithms.genetic.run_ga import run_ga
from vizualization.musclemap_real import MuscleMapReal, BodyGender
from data.data_loader import DataLoader
from planner.planner import Planner


HISTORY_PATHS = {
    "BA": Path("algorithms/bee/results/ba_full_history.json"),
    "GA": Path("algorithms/genetic/results/ga_full_history.json"),
}


def load_history(history_path: Path):
    with open(history_path) as f:
        return json.load(f)


@st.cache_data
def get_muscle_groups_and_popular(limit: int = 7) -> tuple[list[str], list[str]]:
    csv_path = Path("data/exrx_exercises_muscles_clean.csv")
    if not csv_path.exists():
        csv_path = Path("data/exrx_exercises_muscles_with_body_part.csv")

    loader = DataLoader(csv_path)
    exercises_array = loader.exercises()
    planner = Planner(exercises_array, balance_weight=1.0)
    muscle_groups = planner.idx2muscle_group

    frequency: Counter[str] = Counter()
    for exercise in exercises_array:
        for muscles in (exercise.targets, exercise.synergists, exercise.stabilizers):
            for muscle_name in muscles:
                frequency[Planner._normalize_muscle_name(muscle_name)] += 1

    popular_muscles = [name for name, _ in frequency.most_common(limit) if name in set(muscle_groups)]
    return muscle_groups, popular_muscles


def build_muscle_group_weights_from_presets(
    preset_weights: dict[str, float],
    muscle_groups: list[str],
    default_value: float = 1.0,
) -> list[float] | None:
    if not preset_weights:
        return None

    vector = np.full(len(muscle_groups), default_value, dtype=np.float32)
    muscle_idx = {name: idx for idx, name in enumerate(muscle_groups)}
    changed = False

    for muscle_name, weight in preset_weights.items():
        if muscle_name not in muscle_idx:
            continue
        value = float(weight)
        vector[muscle_idx[muscle_name]] = value
        if not np.isclose(value, default_value):
            changed = True

    if not changed:
        return None
    return vector.astype(float).tolist()


def render_vector_param(
    metadata: dict,
    vector_key: str,
    title: str,
    labels: list[str] | None = None,
    default_value: float = 1.0,
) -> None:
    params = params_from_metadata(metadata)
    values = params.get(vector_key)

    if labels is None:
        labels = []

    if not isinstance(values, list):
        return
    if not isinstance(labels, list) or not labels or len(labels) != len(values):
        st.sidebar.caption(f"{title}: vector size {len(values)}")
        return

    vector_df = pd.DataFrame({"muscle": labels, "weight": values})
    changed = vector_df[~np.isclose(vector_df["weight"], default_value)]
    with st.sidebar.expander(title, expanded=False):
        if changed.empty:
            st.caption("All weights use default value.")
        else:
            st.dataframe(changed.sort_values("weight", ascending=False), use_container_width=True, hide_index=True)


def ba_parameters_form(popular_muscles: list[str]) -> dict | None:
    """Sidebar form for BA parameters. Returns dict on submit, None otherwise."""
    with st.sidebar.form("ba_params"):
        st.markdown("**Plan**")
        num_exercises = st.number_input("Number of exercises", 3, 30, 10)
        max_cycles = st.number_input("Number of cycles", 10, 1000, 100, step=10)

        st.markdown("**Population**")
        population_size = st.number_input("Population size", 10, 500, 60, step=10)
        selected_sites = st.number_input("Selected sites (m)", 1, 50, 12)
        elite_sites = st.number_input("Elite sites (e)", 1, 20, 4)
        recruited_elite = st.number_input("Bees / elite (nep)", 1, 100, 16)
        recruited_selected = st.number_input("Bees / selected (nsp)", 1, 50, 8)

        st.markdown("**Neighborhood**")
        mutations = st.number_input("Mutations", 1, 10, 1)
        radius = st.slider("Radius", 0.05, 1.0, 0.35, step=0.05)

        st.markdown("**Planner intensity weights**")
        ba_target_weight = st.number_input("Target weight", 0.0, 10.0, 1.0, step=0.1)
        ba_synergist_weight = st.number_input("Synergist weight", 0.0, 10.0, 1.0, step=0.1)
        ba_stabilizer_weight = st.number_input("Stabilizer weight", 0.0, 10.0, 1.0, step=0.1)

        st.markdown("**Planner muscle group weights (top 7 muscles)**")
        ba_preset_muscle_weights: dict[str, float] = {}
        for idx, muscle_name in enumerate(popular_muscles):
            ba_preset_muscle_weights[muscle_name] = float(
                st.number_input(
                    muscle_name,
                    0.0,
                    10.0,
                    1.0,
                    step=0.1,
                    key=f"ba_muscle_weight_{idx}",
                )
            )

        st.markdown("**Muscle contribution**")
        normalize_helper_muscles = st.checkbox("Normalize helper muscle contributions", value=True)

        st.markdown("**Other**")
        seed = st.number_input("Random seed", 0, 99999, 42)

        submitted = st.form_submit_button("▶️ Run BA", use_container_width=True)
        if not submitted:
            return None
        return {
            "num_exercises": int(num_exercises),
            "max_cycles": int(max_cycles),
            "population_size": int(population_size),
            "selected_sites": int(selected_sites),
            "elite_sites": int(elite_sites),
            "recruited_for_elite": int(recruited_elite),
            "recruited_for_selected": int(recruited_selected),
            "neighborhood_mutations": int(mutations),
            "neighborhood_radius": float(radius),
            "intensity_weights": (
                float(ba_target_weight),
                float(ba_synergist_weight),
                float(ba_stabilizer_weight),
            ),
            "preset_muscle_group_weights": ba_preset_muscle_weights,
            "normalize_helper_muscles": bool(normalize_helper_muscles),
            "random_seed": int(seed),
        }


def ga_parameters_form(popular_muscles: list[str]) -> dict | None:
    """Sidebar form for GA parameters. Returns dict on submit, None otherwise."""
    with st.sidebar.form("ga_params"):
        st.markdown("**Plan**")
        num_exercises = st.number_input("Number of exercises", 3, 30, 10, key="ga_num_exercises")
        max_cycles = st.number_input("Number of cycles", 10, 1000, 100, step=10, key="ga_max_cycles")

        st.markdown("**Population**")
        population_size = st.number_input("Population size", 10, 500, 60, step=10, key="ga_population_size")
        elite_count = st.number_input("Elite count", 0, 100, 4, key="ga_elite_count")

        st.markdown("**Operators**")
        crossover_type = st.selectbox("Crossover type", ["one-point", "two-point", "uniform"], index=0)
        mixing_ratio = st.slider("Mixing ratio (uniform)", 0.0, 1.0, 0.5, step=0.05)
        mutation_chance = st.slider("Mutation chance", 0.0, 1.0, 0.1, step=0.01)

        st.markdown("**Planner intensity weights**")
        ga_target_weight = st.number_input("Target weight", 0.0, 10.0, 1.0, step=0.1, key="ga_target_weight")
        ga_synergist_weight = st.number_input("Synergist weight", 0.0, 10.0, 1.0, step=0.1, key="ga_synergist_weight")
        ga_stabilizer_weight = st.number_input("Stabilizer weight", 0.0, 10.0, 1.0, step=0.1, key="ga_stabilizer_weight")

        st.markdown("**Planner muscle group weights (top 7 muscles)**")
        ga_preset_muscle_weights: dict[str, float] = {}
        for idx, muscle_name in enumerate(popular_muscles):
            ga_preset_muscle_weights[muscle_name] = float(
                st.number_input(
                    muscle_name,
                    0.0,
                    10.0,
                    1.0,
                    step=0.1,
                    key=f"ga_muscle_weight_{idx}",
                )
            )

        st.markdown("**Muscle contribution**")
        normalize_helper_muscles = st.checkbox("Normalize helper muscle contributions", value=True, key="ga_normalize_helper")

        st.markdown("**Other**")
        seed = st.number_input("Random seed", 0, 99999, 42, key="ga_seed")

        submitted = st.form_submit_button("▶️ Run GA", use_container_width=True)
        if not submitted:
            return None
        return {
            "num_exercises": int(num_exercises),
            "max_cycles": int(max_cycles),
            "population_size": int(population_size),
            "elite_count": int(elite_count),
            "crossover_type": str(crossover_type),
            "mixing_ratio": float(mixing_ratio),
            "mutation_chance": float(mutation_chance),
            "intensity_weights": (
                float(ga_target_weight),
                float(ga_synergist_weight),
                float(ga_stabilizer_weight),
            ),
            "preset_muscle_group_weights": ga_preset_muscle_weights,
            "normalize_helper_muscles": bool(normalize_helper_muscles),
            "random_seed": int(seed),
        }


@st.cache_data(show_spinner=False)
def _build_exercise_muscle_matrix(normalize_helper_muscles: bool) -> tuple[np.ndarray, int]:
    """Per-exercise muscle-coverage matrix M[ex_id, muscle_idx] (target+syn+stab summed)."""
    csv_path = Path("data/exrx_exercises_muscles_clean.csv")
    if not csv_path.exists():
        csv_path = Path("data/exrx_exercises_muscles_with_body_part.csv")
    loader = DataLoader(csv_path)
    exercises_array = loader.exercises()
    planner = Planner(exercises_array, balance_weight=1.0, normalize_helper_muscles=normalize_helper_muscles)

    M = np.zeros((len(exercises_array), planner.num_muscle_groups), dtype=np.float32)
    for i, ex in enumerate(exercises_array):
        for muscle in ex.targets:
            M[i, planner.muscle_group2idx[Planner._normalize_muscle_name(muscle)]] += 1.0
        if ex.synergists:
            share = 1.0 / len(ex.synergists) if normalize_helper_muscles else 1.0
            for muscle in ex.synergists:
                M[i, planner.muscle_group2idx[Planner._normalize_muscle_name(muscle)]] += share
        if ex.stabilizers:
            share = 1.0 / len(ex.stabilizers) if normalize_helper_muscles else 1.0
            for muscle in ex.stabilizers:
                M[i, planner.muscle_group2idx[Planner._normalize_muscle_name(muscle)]] += share
    return M, planner.num_muscle_groups


@st.cache_data(show_spinner=False)
def compute_swarm_embedding(history_path_str: str, file_mtime: float, algorithm: str) -> dict | None:
    """PCA-project plans into 2D for both BA and GA.

    BA: indices [:selected_sites] = colony (neighborhood-searched), [selected_sites:] = scouts.
        PCA is fit on the colony only — scouts are uniform-random and would otherwise
        dominate the principal axes.
    GA: the entire population is "the colony" (no scouts; all individuals descend from
        the previous generation via crossover/mutation). PCA is fit on the whole population.
    """
    history = json.load(open(history_path_str))
    cycles = history["cycles"]
    if not cycles or "population" not in cycles[0]:
        return None

    metadata = history["metadata"]
    if algorithm == "BA":
        params = metadata.get("ba_params", {})
        normalize = bool(params.get("normalize_helper_muscles", True))
        elite_count = int(params.get("elite_sites", 4))
        colony_size: int | None = int(params.get("selected_sites", 12))
    else:  # GA
        params = metadata.get("ga_params", {})
        normalize = bool(params.get("normalize_helper_muscles", True))
        elite_count = int(params.get("elite_count", 4))
        colony_size = None  # whole population

    M, _ = _build_exercise_muscle_matrix(normalize)

    populations = np.array([c["population"] for c in cycles], dtype=np.int32)  # (C, P, K)
    costs = np.array([c["population_costs"] for c in cycles], dtype=np.float32)  # (C, P)
    num_cycles, pop_size, _ = populations.shape

    plan_vectors = M[populations].sum(axis=2)  # (C, P, muscles)

    fit_subset = plan_vectors[:, :colony_size, :].reshape(-1, M.shape[1]) if colony_size else plan_vectors.reshape(-1, M.shape[1])
    mean = fit_subset.mean(axis=0, keepdims=True)
    _, sv, vt = np.linalg.svd(fit_subset - mean, full_matrices=False)
    components = vt[:2]

    flat = plan_vectors.reshape(-1, M.shape[1]) - mean
    coords = (flat @ components.T).reshape(num_cycles, pop_size, 2)
    explained = (sv[:2] ** 2 / (sv ** 2).sum()).tolist()

    best_idx_per_cycle = np.argmin(costs, axis=1)
    best_costs = costs[np.arange(num_cycles), best_idx_per_cycle]
    if colony_size:
        centroid_positions = coords[:, :colony_size, :].mean(axis=1)
    else:
        centroid_positions = coords.mean(axis=1)

    global_best_positions = np.empty((num_cycles, 2), dtype=np.float32)
    global_best_costs = np.empty(num_cycles, dtype=np.float32)
    record_cost = np.inf
    record_pos = coords[0, int(best_idx_per_cycle[0])]
    for c in range(num_cycles):
        bi = int(best_idx_per_cycle[c])
        if costs[c, bi] < record_cost:
            record_cost = float(costs[c, bi])
            record_pos = coords[c, bi]
        global_best_positions[c] = record_pos
        global_best_costs[c] = record_cost

    return {
        "coords": coords.tolist(),
        "costs": costs.tolist(),
        "best_idx": best_idx_per_cycle.tolist(),
        "best_costs": best_costs.tolist(),
        "centroid_positions": centroid_positions.tolist(),
        "global_best_positions": global_best_positions.tolist(),
        "global_best_costs": global_best_costs.tolist(),
        "explained": explained,
        "elite_count": elite_count,
        "colony_size": colony_size if colony_size is not None else pop_size,
        "has_scouts": colony_size is not None and colony_size < pop_size,
        "algorithm": algorithm,
    }


def render_swarm_trajectory(history: dict, history_path: Path, algorithm: str) -> None:
    embedding = compute_swarm_embedding(str(history_path), history_path.stat().st_mtime, algorithm)
    if embedding is None:
        st.info(f"Run {algorithm} to populate the trajectory view — re-run from the sidebar so per-cycle populations are saved.")
        return

    coords = np.array(embedding["coords"])
    costs = np.array(embedding["costs"])
    centroid_positions = np.array(embedding["centroid_positions"])
    global_best_positions = np.array(embedding["global_best_positions"])
    global_best_costs = np.array(embedding["global_best_costs"])
    ev1, ev2 = embedding["explained"]
    elite_count = int(embedding["elite_count"])
    colony_size = int(embedding["colony_size"])
    has_scouts = bool(embedding["has_scouts"])
    num_cycles, pop_size, _ = coords.shape

    rest_label = "selected" if has_scouts else "offspring"
    centroid_label = "colony centroid trail" if algorithm == "BA" else "population centroid trail"

    colony_costs = costs[:, :colony_size]
    cmin, cmax = float(colony_costs.min()), float(colony_costs.max())

    frames = []
    for c in range(num_cycles):
        elite_xy = coords[c, :elite_count]
        rest_xy = coords[c, elite_count:colony_size]
        scout_xy = coords[c, colony_size:] if has_scouts else None

        elite_c = costs[c, :elite_count]
        rest_c = costs[c, elite_count:colony_size]

        frame_data = []

        if has_scouts:
            frame_data.append(go.Scatter(
                x=scout_xy[:, 0], y=scout_xy[:, 1],
                mode="markers",
                marker=dict(size=5, color="rgba(160,160,200,0.45)", line=dict(width=0)),
                hoverinfo="skip",
                showlegend=(c == 0), name="scouts",
            ))

        frame_data += [
            go.Scatter(
                x=centroid_positions[: c + 1, 0], y=centroid_positions[: c + 1, 1],
                mode="lines",
                line=dict(color="rgba(0, 220, 255, 0.55)", width=2, dash="dot"),
                hoverinfo="skip",
                showlegend=(c == 0), name=centroid_label,
            ),
            go.Scatter(
                x=global_best_positions[: c + 1, 0], y=global_best_positions[: c + 1, 1],
                mode="lines",
                line=dict(color="rgba(255, 195, 0, 0.9)", width=3.2, shape="hv"),
                hoverinfo="skip",
                showlegend=(c == 0), name="global best trail",
            ),
            go.Scatter(
                x=rest_xy[:, 0], y=rest_xy[:, 1],
                mode="markers",
                marker=dict(size=12, color=rest_c, colorscale="Viridis_r",
                            cmin=cmin, cmax=cmax,
                            line=dict(color="rgba(255,255,255,0.6)", width=0.8),
                            showscale=False),
                customdata=rest_c.reshape(-1, 1),
                hovertemplate=rest_label + " · cost %{customdata[0]:.3f}<extra></extra>",
                showlegend=(c == 0), name=rest_label,
            ),
            go.Scatter(
                x=elite_xy[:, 0], y=elite_xy[:, 1],
                mode="markers",
                marker=dict(size=18, color=elite_c, colorscale="Viridis_r",
                            cmin=cmin, cmax=cmax,
                            symbol="hexagon",
                            line=dict(color="white", width=1.4),
                            colorbar=dict(title="Cost", len=0.7, thickness=14)),
                customdata=elite_c.reshape(-1, 1),
                hovertemplate="elite · cost %{customdata[0]:.3f}<extra></extra>",
                showlegend=(c == 0), name="elite",
            ),
            go.Scatter(
                x=[global_best_positions[c, 0]], y=[global_best_positions[c, 1]],
                mode="markers",
                marker=dict(symbol="star", size=26, color="gold",
                            line=dict(color="black", width=1.4)),
                hovertemplate=f"global best so far<br>cost {global_best_costs[c]:.3f}<extra></extra>",
                showlegend=(c == 0), name="global best",
            ),
        ]

        frames.append(go.Frame(name=str(c + 1), data=frame_data))

    sx_min, sx_max = float(np.percentile(coords[..., 0], 1)), float(np.percentile(coords[..., 0], 99))
    sy_min, sy_max = float(np.percentile(coords[..., 1], 1)), float(np.percentile(coords[..., 1], 99))
    pad_x = max((sx_max - sx_min) * 0.08, 0.5)
    pad_y = max((sy_max - sy_min) * 0.08, 0.5)
    x_range = [sx_min - pad_x, sx_max + pad_x]
    y_range = [sy_min - pad_y, sy_max + pad_y]

    title_subject = "Bee Swarm" if algorithm == "BA" else "Population"
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=f"{title_subject} Trajectory in Fitness Landscape — PCA on muscle-coverage  ·  PC1 {ev1*100:.1f}%, PC2 {ev2*100:.1f}%",
        xaxis=dict(title="PC1", range=x_range, zeroline=False),
        yaxis=dict(title="PC2", range=y_range, zeroline=False, scaleanchor="x", scaleratio=1),
        height=640,
        margin=dict(l=10, r=10, t=70, b=110),
        plot_bgcolor="rgba(20,22,30,1)",
        paper_bgcolor="rgba(20,22,30,1)",
        font=dict(color="#e6e6e6"),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0, y=-0.18, xanchor="left", yanchor="top",
            pad=dict(r=8, t=4),
            showactive=False,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=120, redraw=True),
                                      transition=dict(duration=60),
                                      fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate", transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.13, y=-0.05, len=0.87,
            currentvalue=dict(prefix="Cycle: ", font=dict(size=14, color="#ffd54a")),
            steps=[dict(method="animate", label=str(c + 1),
                        args=[[str(c + 1)], dict(mode="immediate",
                                                  frame=dict(duration=0, redraw=True),
                                                  transition=dict(duration=0))])
                   for c in range(num_cycles)],
        )],
    )

    st.plotly_chart(fig, use_container_width=True)
    if has_scouts:
        st.caption(
            "⬢ hexagon — elite site  ·  ● circle — selected site  ·  · dot — scout  ·  "
            "★ star — global best  ·  ┄ cyan dotted — colony centroid trail  ·  "
            "─ gold step — global-best trail  ·  color — cost (brighter = better)"
        )
    else:
        st.caption(
            "⬢ hexagon — elite (preserved across generations)  ·  ● circle — offspring  ·  "
            "★ star — global best  ·  ┄ cyan dotted — population centroid trail  ·  "
            "─ gold step — global-best trail  ·  color — cost (brighter = better)"
        )


def params_from_metadata(metadata: dict) -> dict:
    for key, value in metadata.items():
        if key.endswith("_params") and isinstance(value, dict):
            return value
    return {}


def main():
    st.set_page_config(layout="wide", page_title="Workout Algorithm Dashboard")
    st.title("Workout Algorithm Dashboard")
    muscle_groups, popular_muscles = get_muscle_groups_and_popular(limit=7)

    st.sidebar.header("⚙️ Algorithm parameters")
    algorithm = st.sidebar.radio("Algorithm", ["BA", "GA"], horizontal=True)
    history_path = HISTORY_PATHS[algorithm]

    if algorithm == "BA":
        new_params = ba_parameters_form(popular_muscles)
        if new_params is not None and new_params["elite_sites"] > new_params["selected_sites"]:
            st.sidebar.error("elite_sites must be ≤ selected_sites")
        else:
            if new_params is not None:
                new_params["muscle_group_weights"] = build_muscle_group_weights_from_presets(
                    new_params.pop("preset_muscle_group_weights"),
                    muscle_groups,
                )
            if new_params is not None:
                with st.spinner("Running Bees Algorithm…"):
                    run_ba(**new_params)
                st.sidebar.success("Done - results reloaded.")
    else:
        new_params = ga_parameters_form(popular_muscles)
        if new_params is not None and new_params["elite_count"] >= new_params["population_size"]:
            st.sidebar.error("elite_count must be < population_size")
        else:
            if new_params is not None:
                new_params["muscle_group_weights"] = build_muscle_group_weights_from_presets(
                    new_params.pop("preset_muscle_group_weights"),
                    muscle_groups,
                )
            if new_params is not None:
                with st.spinner("Running Genetic Algorithm…"):
                    run_ga(**new_params)
                st.sidebar.success("Done - results reloaded.")

    if not history_path.exists():
        st.warning("No results. Run the algorithm in the left panel.")
        return

    history = load_history(history_path)
    cycles = history["cycles"]
    metadata = history["metadata"]
    algorithm_params = params_from_metadata(metadata)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Last run:**")
    for param, value in algorithm_params.items():
        if param == "muscle_group_weights":
            continue
        st.sidebar.text(f"{param}: {value}")
    render_vector_param(
        metadata,
        "muscle_group_weights",
        "Muscle Group Weights",
        labels=history.get("muscle_groups", []),
        default_value=1.0,
    )

    gender = st.sidebar.radio("Body type", ["Male", "Female"], horizontal=True)
    body_gender = BodyGender.MALE if gender == "Male" else BodyGender.FEMALE

    cycle_idx = st.sidebar.slider("Select cycle", 0, len(cycles) - 1, len(cycles) - 1)
    cycle_data = cycles[cycle_idx]

    # Convergence graph
    st.header("Convergence Graph")
    fig_cost, ax = plt.subplots(figsize=(14, 5))
    costs = [c["best_cost"] for c in cycles]
    ax.plot(range(1, len(costs) + 1), costs, "b-", linewidth=2.5, label="Best Cost")
    ax.axvline(x=cycle_idx + 1, color="red", linestyle="--", alpha=0.7, linewidth=2)
    ax.fill_between(range(1, len(costs) + 1), costs, alpha=0.2)
    ax.set_xlabel("Cycle", fontsize=12, weight="bold")
    ax.set_ylabel("Cost", fontsize=12, weight="bold")
    ax.set_title(f"{algorithm} Convergence Over Time", fontsize=14, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    st.pyplot(fig_cost)

    header_emoji = "🐝" if algorithm == "BA" else "🧬"
    header_label = "Bee Swarm Trajectory" if algorithm == "BA" else "Population Trajectory"
    st.header(f"{header_emoji} {header_label}")
    render_swarm_trajectory(history, history_path, algorithm)

    # Cycle info
    st.header(f"Cycle {cycle_data['cycle']} Statistics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Best Cost", f"{cycle_data['best_cost']:.4f}")
    col2.metric("Avg Cost", f"{cycle_data['avg_cost']:.4f}")
    col3.metric("Min Cost", f"{cycle_data['min_cost']:.4f}")
    col4.metric("Population", cycle_data["population_size"])
    col5.metric("Cycle", f"{cycle_data['cycle']} / {len(cycles)}")
    cum_evals = cycle_data.get("fitness_evals")
    total_evals = metadata.get("total_fitness_evals")
    if cum_evals is not None:
        label = "Cost evals (up to cycle)"
        value = f"{cum_evals:,}"
        delta = f"total: {total_evals:,}" if total_evals is not None else None
        col6.metric(label, value, delta=delta, delta_color="off")
    elif total_evals is not None:
        col6.metric("Cost evals (total)", f"{total_evals:,}")

    # Body map with intensity - Real MuscleMap version
    st.header("Muscle Activation Map - Real MuscleMap SVG Paths")
    body_model = MuscleMapReal(gender=body_gender, figsize=(14, 8))
    muscle_data = cycle_data.get("intensity_by_muscle", [])
    body_model.apply_muscle_intensity(muscle_data)
    body_model.fig.suptitle(
        f"Cycle {cycle_data['cycle']} - Muscle Activation Heatmap ({gender})",
        fontsize=16, weight="bold"
    )
    st.pyplot(body_model.fig)

    # Exercise plan
    st.header("Selected Exercises")
    names = cycle_data["exercise_names"]
    best_plan = cycle_data.get("best_plan", [])
    urls = cycle_data.get("exercise_urls", [""] * len(names))
    st.dataframe(
        pd.DataFrame({
            "#": range(1, len(names) + 1),
            "Exercise ID": best_plan,
            "Exercise": names,
            "Link": urls,
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn(
                "Link", display_text="ExRx →", help="Open exercise description on ExRx.net"
            ),
        },
    )

    # Muscle intensity details
    st.header("Muscle Activation Details")
    muscle_df = pd.DataFrame(cycle_data["intensity_by_muscle"])
    if not muscle_df.empty:
        muscle_df = muscle_df.sort_values("total_intensity", ascending=False)
        st.dataframe(
            muscle_df[["muscle", "target_count", "synergist_count", "stabilizer_count", "total_intensity"]].head(20),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
