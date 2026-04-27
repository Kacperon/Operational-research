#!/usr/bin/env python3
"""Interactive dashboard - Bees Algorithm + MuscleMap visualization."""

import json
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from bee.run_ba import run_ba
from vizualization.musclemap_real import MuscleMapReal, BodyGender


HISTORY_PATH = Path("bee/results/ba_full_history.json")


def load_history():
    with open(HISTORY_PATH) as f:
        return json.load(f)


def ba_parameters_form() -> dict | None:
    """Sidebar form for BA parameters. Returns dict on submit, None otherwise."""
    with st.sidebar.form("ba_params"):
        st.markdown("**Plan**")
        num_exercises = st.number_input("Liczba ćwiczeń", 3, 30, 10)
        max_cycles = st.number_input("Liczba cykli", 10, 1000, 100, step=10)

        st.markdown("**Populacja**")
        population_size = st.number_input("Rozmiar populacji", 10, 500, 60, step=10)
        selected_sites = st.number_input("Selected sites (m)", 1, 50, 12)
        elite_sites = st.number_input("Elite sites (e)", 1, 20, 4)
        recruited_elite = st.number_input("Pszczoły / elite (nep)", 1, 100, 16)
        recruited_selected = st.number_input("Pszczoły / selected (nsp)", 1, 50, 8)

        st.markdown("**Sąsiedztwo**")
        mutations = st.number_input("Mutacje", 1, 10, 1)
        radius = st.slider("Promień", 0.05, 1.0, 0.35, step=0.05)

        st.markdown("**Inne**")
        seed = st.number_input("Ziarno losowe", 0, 99999, 42)

        submitted = st.form_submit_button("▶️ Uruchom BA", use_container_width=True)
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
            "random_seed": int(seed),
        }


def main():
    st.set_page_config(layout="wide", page_title="BA Workout Dashboard")
    st.title("🏋️ Bees Algorithm Workout Dashboard")

    st.sidebar.header("⚙️ Parametry algorytmu")
    new_params = ba_parameters_form()
    if new_params is not None:
        if new_params["elite_sites"] > new_params["selected_sites"]:
            st.sidebar.error("elite_sites musi być ≤ selected_sites")
        else:
            with st.spinner("Uruchamiam Bees Algorithm…"):
                run_ba(**new_params)
            st.sidebar.success("Gotowe — wyniki przeładowane.")

    if not HISTORY_PATH.exists():
        st.warning("Brak wyników. Uruchom algorytm w panelu po lewej.")
        return

    history = load_history()
    cycles = history["cycles"]
    metadata = history["metadata"]

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Ostatni przebieg:**")
    for param, value in metadata["ba_params"].items():
        st.sidebar.text(f"{param}: {value}")

    gender = st.sidebar.radio("Sylwetka", ["Male", "Female"], horizontal=True)
    body_gender = BodyGender.MALE if gender == "Male" else BodyGender.FEMALE

    cycle_idx = st.sidebar.slider("Wybierz cykl", 0, len(cycles) - 1, len(cycles) - 1)
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
    ax.set_title("Algorithm Convergence Over Time", fontsize=14, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    st.pyplot(fig_cost)

    # Cycle info
    st.header(f"Cycle {cycle_data['cycle']} Statistics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Best Cost", f"{cycle_data['best_cost']:.4f}")
    col2.metric("Avg Cost", f"{cycle_data['avg_cost']:.4f}")
    col3.metric("Min Cost", f"{cycle_data['min_cost']:.4f}")
    col4.metric("Population", cycle_data["population_size"])
    col5.metric("Cykl", f"{cycle_data['cycle']} / {len(cycles)}")
    cum_evals = cycle_data.get("fitness_evals")
    total_evals = metadata.get("total_fitness_evals")
    if cum_evals is not None:
        label = "Wywołania kosztu (do cyklu)"
        value = f"{cum_evals:,}"
        delta = f"łącznie: {total_evals:,}" if total_evals is not None else None
        col6.metric(label, value, delta=delta, delta_color="off")
    elif total_evals is not None:
        col6.metric("Wywołania kosztu (łącznie)", f"{total_evals:,}")

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
    urls = cycle_data.get("exercise_urls", [""] * len(names))
    st.dataframe(
        pd.DataFrame({
            "#": range(1, len(names) + 1),
            "Exercise": names,
            "Link": urls,
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn(
                "Link", display_text="ExRx →", help="Otwórz opis ćwiczenia na ExRx.net"
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
