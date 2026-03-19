from __future__ import annotations

import argparse
import random
import re
import textwrap
import webbrowser
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch
from matplotlib.widgets import Button


ROLE_COLUMNS = {
	"Target": "Target",
	"Synergists": "Synergists",
	"Dynamic Stabilizers": "Dynamic Stabilizers",
	"Stabilizers": "Stabilizers",
	"Antagonist Stabilizers": "Antagonist Stabilizers",
}

ROLE_WEIGHTS = {
	"Target": 1.0,
	"Synergists": 0.7,
	"Dynamic Stabilizers": 0.3,
	"Stabilizers": 0.3,
	"Antagonist Stabilizers": 0.3,
}


def normalize_text(value: str) -> str:
	value = (value or "").strip().lower()
	value = re.sub(r"\s+", " ", value)
	return value


def parse_muscle_list(value: object) -> list[str]:
	if value is None:
		return []

	text = str(value).strip()
	if not text or text.lower() in {"nan", "none"}:
		return []
	if "no significant stabilizers" in text.lower():
		return []

	parts = [p.strip() for p in text.split(";")]
	return [p for p in parts if p]


MUSCLE_TO_REGIONS = {
	# Neck / shoulders / upper back
	"trapezius, upper": ["back:traps", "front:neck"],
	"trapezius, middle": ["back:traps"],
	"levator scapulae": ["back:traps"],
	"deltoid": ["front:shoulders", "back:rear_shoulders"],
	"deltoid, anterior": ["front:shoulders"],
	"deltoid, lateral": ["front:shoulders", "back:rear_shoulders"],
	"deltoid, posterior": ["back:rear_shoulders"],
	# Chest / arms
	"pectoralis major": ["front:chest"],
	"biceps brachii": ["front:biceps"],
	"brachialis": ["front:biceps"],
	"brachioradialis": ["front:forearms"],
	"triceps brachii": ["back:triceps"],
	# Core
	"rectus abdominis": ["front:abs"],
	"obliques": ["front:obliques", "back:obliques_back"],
	"quadratus lumborum": ["back:lower_back"],
	"erector spinae": ["back:lower_back"],
	# Back
	"latissimus dorsi": ["back:lats"],
	"rhomboids": ["back:lats"],
	# Glutes / legs
	"gluteus maximus": ["back:glutes"],
	"gluteus medius": ["back:glutes", "front:hips"],
	"gluteus minimus": ["back:glutes", "front:hips"],
	"quadriceps": ["front:quads"],
	"vastus lateralis": ["front:quads"],
	"vastus medialis": ["front:quads"],
	"vastus intermedius": ["front:quads"],
	"rectus femoris": ["front:quads"],
	"adductor magnus": ["front:adductors"],
	"adductor longus": ["front:adductors"],
	"adductor brevis": ["front:adductors"],
	"hamstrings": ["back:hamstrings"],
	"biceps femoris": ["back:hamstrings"],
	"semitendinosus": ["back:hamstrings"],
	"semimembranosus": ["back:hamstrings"],
	"gastrocnemius": ["front:calves", "back:calves"],
	"soleus": ["back:calves", "front:calves"],
	"tibialis anterior": ["front:shins"],
}


BODY_PART_TO_REGIONS = {
	"calves": ["front:calves", "back:calves"],
	"shoulders": ["front:shoulders", "back:rear_shoulders"],
	"chest": ["front:chest"],
	"back": ["back:lats", "back:traps", "back:lower_back"],
	"legs": ["front:quads", "back:hamstrings", "back:glutes", "front:calves", "back:calves"],
	"abdominals": ["front:abs", "front:obliques"],
	"glutes": ["back:glutes"],
	"arms": ["front:biceps", "back:triceps", "front:forearms"],
}


def resolve_data_file(explicit_path: str | None) -> Path:
	if explicit_path:
		path = Path(explicit_path).expanduser().resolve()
		if not path.exists():
			raise FileNotFoundError(f"Nie znaleziono pliku CSV: {path}")
		return path

	repo_root = Path(__file__).resolve().parents[1]
	candidates = [
		repo_root / "data" / "exrx_exercises_muscles_clean.csv",
	]
	for candidate in candidates:
		if candidate.exists():
			return candidate

	raise FileNotFoundError("Nie znaleziono żadnego pliku CSV z ćwiczeniami w katalogu data/.")


class ExerciseBodyVisualizer:
	def __init__(self, df: pd.DataFrame):
		self.df = df.reset_index(drop=True)
		self.idx = 0
		self.current_url = ""

		self.fig = plt.figure(figsize=(14, 8))
		gs = self.fig.add_gridspec(
			nrows=3,
			ncols=4,
			height_ratios=[14, 1.5, 1.5],
			width_ratios=[1, 1, 1, 1],
		)

		self.ax_front = self.fig.add_subplot(gs[0, 0])
		self.ax_back = self.fig.add_subplot(gs[0, 1])
		self.ax_info = self.fig.add_subplot(gs[0, 2])
		self.ax_prev = self.fig.add_subplot(gs[1, 0])
		self.ax_next = self.fig.add_subplot(gs[1, 1])
		self.ax_random = self.fig.add_subplot(gs[1, 2])
		self.ax_open = self.fig.add_subplot(gs[1, 3])

		self.front_regions = self._draw_human(self.ax_front, view="front")
		self.back_regions = self._draw_human(self.ax_back, view="back")

		self.ax_info.axis("off")
		self.info_text = self.ax_info.text(
			0,
			1,
			"",
			va="top",
			fontsize=11,
			wrap=True,
			transform=self.ax_info.transAxes,
		)

		self.btn_prev = Button(self.ax_prev, "Poprzednie")
		self.btn_next = Button(self.ax_next, "Następne")
		self.btn_random = Button(self.ax_random, "Losuj")
		self.btn_open = Button(self.ax_open, "Otwórz opis ćwiczenia")

		self.btn_prev.on_clicked(self._on_prev)
		self.btn_next.on_clicked(self._on_next)
		self.btn_random.on_clicked(self._on_random)
		self.btn_open.on_clicked(self._on_open_url)

		self.fig.canvas.mpl_connect("key_press_event", self._on_key)
		self.update_view()

	def _draw_human(self, ax, view: str):
		ax.set_xlim(0, 1)
		ax.set_ylim(0, 2)
		ax.set_aspect("equal")
		ax.axis("off")

		ax.text(0.5, 1.95, "PRZÓD" if view == "front" else "TYŁ", ha="center", fontsize=13, weight="bold")

		# Base silhouette
		skin = "#e5e7eb"
		edge = "#9ca3af"
		ax.add_patch(Circle((0.5, 1.78), 0.09, facecolor=skin, edgecolor=edge, lw=1.2))
		ax.add_patch(FancyBboxPatch((0.39, 1.05), 0.22, 0.62, boxstyle="round,pad=0.02,rounding_size=0.06", facecolor=skin, edgecolor=edge, lw=1.2))
		ax.add_patch(FancyBboxPatch((0.26, 1.12), 0.12, 0.48, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor=skin, edgecolor=edge, lw=1.2))
		ax.add_patch(FancyBboxPatch((0.62, 1.12), 0.12, 0.48, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor=skin, edgecolor=edge, lw=1.2))
		ax.add_patch(FancyBboxPatch((0.41, 0.50), 0.08, 0.55, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=skin, edgecolor=edge, lw=1.2))
		ax.add_patch(FancyBboxPatch((0.51, 0.50), 0.08, 0.55, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=skin, edgecolor=edge, lw=1.2))
		ax.add_patch(FancyBboxPatch((0.41, 0.12), 0.08, 0.35, boxstyle="round,pad=0.01,rounding_size=0.03", facecolor=skin, edgecolor=edge, lw=1.2))
		ax.add_patch(FancyBboxPatch((0.51, 0.12), 0.08, 0.35, boxstyle="round,pad=0.01,rounding_size=0.03", facecolor=skin, edgecolor=edge, lw=1.2))

		# Highlightable regions
		regions = {}

		def reg(key: str, patch):
			patch.set_facecolor("#fee2e2")
			patch.set_edgecolor("#ef4444")
			patch.set_alpha(0.07)
			patch.set_linewidth(1.0)
			ax.add_patch(patch)
			regions[key] = patch

		if view == "front":
			reg("front:neck", Ellipse((0.5, 1.64), 0.10, 0.08))
			reg("front:shoulders", Ellipse((0.5, 1.56), 0.36, 0.16))
			reg("front:chest", Ellipse((0.5, 1.42), 0.24, 0.18))
			reg("front:abs", FancyBboxPatch((0.44, 1.12), 0.12, 0.22, boxstyle="round,pad=0.01,rounding_size=0.03"))
			reg("front:obliques", Ellipse((0.5, 1.18), 0.24, 0.18))
			reg("front:biceps", Ellipse((0.32, 1.30), 0.09, 0.16))
			reg("front:biceps_r", Ellipse((0.68, 1.30), 0.09, 0.16))
			reg("front:forearms", Ellipse((0.31, 1.02), 0.08, 0.16))
			reg("front:forearms_r", Ellipse((0.69, 1.02), 0.08, 0.16))
			reg("front:hips", Ellipse((0.5, 0.97), 0.22, 0.12))
			reg("front:quads", FancyBboxPatch((0.42, 0.62), 0.16, 0.34, boxstyle="round,pad=0.02,rounding_size=0.04"))
			reg("front:adductors", Ellipse((0.5, 0.68), 0.10, 0.20))
			reg("front:shins", FancyBboxPatch((0.44, 0.24), 0.12, 0.22, boxstyle="round,pad=0.02,rounding_size=0.03"))
			reg("front:calves", FancyBboxPatch((0.43, 0.16), 0.14, 0.22, boxstyle="round,pad=0.02,rounding_size=0.04"))
		else:
			reg("back:traps", Ellipse((0.5, 1.54), 0.28, 0.18))
			reg("back:rear_shoulders", Ellipse((0.5, 1.46), 0.34, 0.15))
			reg("back:lats", Ellipse((0.5, 1.30), 0.30, 0.24))
			reg("back:triceps", Ellipse((0.32, 1.24), 0.09, 0.17))
			reg("back:triceps_r", Ellipse((0.68, 1.24), 0.09, 0.17))
			reg("back:lower_back", FancyBboxPatch((0.44, 1.05), 0.12, 0.18, boxstyle="round,pad=0.02,rounding_size=0.03"))
			reg("back:obliques_back", Ellipse((0.5, 1.08), 0.23, 0.16))
			reg("back:glutes", Ellipse((0.5, 0.92), 0.22, 0.16))
			reg("back:hamstrings", FancyBboxPatch((0.42, 0.60), 0.16, 0.34, boxstyle="round,pad=0.02,rounding_size=0.04"))
			reg("back:calves", FancyBboxPatch((0.43, 0.18), 0.14, 0.28, boxstyle="round,pad=0.02,rounding_size=0.05"))

		return regions

	def _lookup_regions(self, muscle_name: str, body_part: str) -> list[str]:
		key = normalize_text(muscle_name)
		direct = MUSCLE_TO_REGIONS.get(key)
		if direct:
			return direct

		# fallback by broad body part, if muscle mapping is missing
		part_key = normalize_text(body_part)
		return BODY_PART_TO_REGIONS.get(part_key, [])

	def _compute_scores(self, row: pd.Series) -> tuple[dict[str, float], dict[str, list[str]]]:
		scores = defaultdict(float)
		role_muscles: dict[str, list[str]] = {}

		body_part = str(row.get("body_part", ""))
		for role, column in ROLE_COLUMNS.items():
			muscles = parse_muscle_list(row.get(column, ""))
			role_muscles[role] = muscles
			weight = ROLE_WEIGHTS[role]
			for muscle in muscles:
				for region in self._lookup_regions(muscle, body_part):
					scores[region] += weight

		# if no specific muscle got mapped, at least show body part region
		if not scores:
			for region in BODY_PART_TO_REGIONS.get(normalize_text(body_part), []):
				scores[region] += 0.8

		# mirror left/right where region has _r variant
		mirrored = dict(scores)
		for region, value in list(scores.items()):
			if region.endswith("_r"):
				base = region[:-2]
				mirrored[base] = max(mirrored.get(base, 0), value)
			else:
				right = f"{region}_r"
				if right in self.front_regions or right in self.back_regions:
					mirrored[right] = max(mirrored.get(right, 0), value)

		return mirrored, role_muscles

	def _apply_region_colors(self, scores: dict[str, float]):
		max_score = max(scores.values(), default=1.0)
		cmap = cm.get_cmap("YlOrRd")

		for region_map in (self.front_regions, self.back_regions):
			for region_name, patch in region_map.items():
				score = scores.get(region_name, 0.0)
				if score <= 0:
					patch.set_alpha(0.07)
					patch.set_facecolor("#fee2e2")
					patch.set_edgecolor("#ef4444")
				else:
					intensity = min(score / max_score, 1.0)
					patch.set_facecolor(cmap(0.25 + 0.75 * intensity))
					patch.set_edgecolor("#7f1d1d")
					patch.set_alpha(0.35 + 0.6 * intensity)

	def update_view(self):
		row = self.df.iloc[self.idx]
		scores, role_muscles = self._compute_scores(row)
		self._apply_region_colors(scores)

		exercise_name = str(row.get("exercise_name", "Brak nazwy"))
		body_part = str(row.get("body_part", "nieznana grupa"))
		url = str(row.get("exercise_url", "")).strip()
		self.current_url = url

		self.fig.suptitle(
			f"{self.idx + 1}/{len(self.df)} • {exercise_name} • {body_part}",
			fontsize=14,
			weight="bold",
		)

		lines = [
			f"Ćwiczenie: {exercise_name}",
			f"Body part: {body_part}",
			"",
			"Zaangażowanie mięśni:",
		]

		for role in ROLE_COLUMNS:
			muscles = role_muscles.get(role, [])
			if muscles:
				lines.append(f"• {role}: {', '.join(muscles[:5])}{' …' if len(muscles) > 5 else ''}")

		if url:
			short_url = url if len(url) < 80 else url[:77] + "..."
			lines.extend(["", f"URL: {short_url}"])

		# Wrap long lines so they stay inside the info panel.
		wrapped_lines: list[str] = []
		for line in lines:
			if not line:
				wrapped_lines.append("")
			else:
				wrapped_lines.extend(textwrap.wrap(line, width=56) or [""])

		self.info_text.set_text("\n".join(wrapped_lines))
		self.fig.canvas.draw_idle()

	def _on_prev(self, _event):
		self.idx = (self.idx - 1) % len(self.df)
		self.update_view()

	def _on_next(self, _event):
		self.idx = (self.idx + 1) % len(self.df)
		self.update_view()

	def _on_open_url(self, _event):
		if self.current_url:
			webbrowser.open(self.current_url, new=2)

	def _on_random(self, _event):
		self.idx = random.randrange(len(self.df))
		self.update_view()

	def _on_key(self, event):
		if event.key in {"right", "d", "n"}:
			self._on_next(event)
		elif event.key in {"left", "a", "p"}:
			self._on_prev(event)
		elif event.key in {"r"}:
			self._on_random(event)
		elif event.key in {"o", "enter"}:
			self._on_open_url(event)


def load_data(csv_path: str | None, body_part_filter: str | None, contains: str | None) -> pd.DataFrame:
	path = resolve_data_file(csv_path)
	df = pd.read_csv(path)

	required = {"exercise_name", "exercise_url", "body_part"}.union(set(ROLE_COLUMNS.values()))
	missing = [col for col in required if col not in df.columns]
	if missing:
		raise ValueError(f"Brakuje kolumn w CSV: {missing}")

	if body_part_filter:
		df = df[df["body_part"].str.contains(body_part_filter, case=False, na=False)]
	if contains:
		df = df[df["exercise_name"].str.contains(contains, case=False, na=False)]

	if df.empty:
		raise ValueError("Brak danych po zastosowaniu filtrów.")

	return df


def main():
	parser = argparse.ArgumentParser(
		description="Wizualizacja ćwiczeń na modelu człowieka (z linkami do opisu)."
	)
	parser.add_argument("--csv", help="Ścieżka do pliku CSV.")
	parser.add_argument("--body-part", help="Filtr po body_part, np. Calves")
	parser.add_argument("--contains", help="Filtr po fragmencie nazwy ćwiczenia")
	args = parser.parse_args()

	df = load_data(args.csv, args.body_part, args.contains)
	ExerciseBodyVisualizer(df)
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
