"""Exercise body visualizer using anatomical SVG muscle model.

Modes:
  browse  – navigate exercises one by one (role-based coloring)
  plan    – aggregate heatmap across a list of exercises (solver output)
"""
from __future__ import annotations

import argparse
import random
import re
import webbrowser
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.widgets import Button
from svgpath2mpl import parse_path

from config import (
	BODY_PART_TO_SVG,
	CSV_TO_ROLE,
	DISPLAY_ROLES,
	GROUP_SPLITS,
	INACTIVE_ALPHA,
	INACTIVE_EDGE,
	INACTIVE_FACE,
	MUSCLE_SVG_IDS,
	MUSCLE_TO_SVG,
	ROLE_COLORS,
	ROLE_COLUMNS,
	ROLE_EDGE_COLOR,
	ROLE_PRIORITY,
	ROLE_WEIGHTS,
	STRUCTURE_SVG_IDS,
)

# ─── Paths ────────────────────────────────────────────────────────

SVG_FILE = Path(__file__).parent / "Muscles-simplified.svg"
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "exrx_exercises_muscles_clean.csv"

# ─── Helpers ──────────────────────────────────────────────────────


def normalize_muscle(name: str) -> str:
	name = name.strip().lower()
	name = re.sub(r"\s+", " ", name)
	name = re.sub(r"\s*\(.*?\)", "", name).strip()
	return name


def parse_muscle_list(value: object) -> list[str]:
	if value is None:
		return []
	text = str(value).strip()
	if not text or text.lower() in {"nan", "none"}:
		return []
	if "no significant stabilizers" in text.lower():
		return []
	return [p.strip() for p in text.split(";") if p.strip()]


def resolve_muscle(raw_name: str) -> list[str]:
	name = normalize_muscle(raw_name)
	if name in MUSCLE_TO_SVG:
		return MUSCLE_TO_SVG[name]
	best, best_regions = "", []
	for key, regions in MUSCLE_TO_SVG.items():
		if (name.startswith(key) or key.startswith(name)) and len(key) > len(best):
			best, best_regions = key, regions
	return best_regions


def resolve_body_parts(body_part_str: str) -> list[str]:
	regions: list[str] = []
	for part in body_part_str.split(";"):
		part = part.strip().lower()
		regions.extend(BODY_PART_TO_SVG.get(part, []))
	return regions


def _merged_muscles(row: pd.Series) -> dict[str, list[str]]:
	"""Collect muscles per display role, merging CSV columns."""
	merged: dict[str, list[str]] = {r: [] for r in DISPLAY_ROLES}
	for csv_col in ROLE_COLUMNS:
		display_role = CSV_TO_ROLE[csv_col]
		for m in parse_muscle_list(row.get(csv_col, "")):
			if m not in merged[display_role]:
				merged[display_role].append(m)
	return merged


# ─── Scoring ──────────────────────────────────────────────────────


def compute_exercise_scores(row: pd.Series) -> dict[str, float]:
	"""Weighted score per SVG region (for heatmap / plan mode)."""
	scores: dict[str, float] = defaultdict(float)
	body_part = str(row.get("body_part", ""))
	for csv_col in ROLE_COLUMNS:
		display_role = CSV_TO_ROLE[csv_col]
		weight = ROLE_WEIGHTS[display_role]
		muscles = parse_muscle_list(row.get(csv_col, ""))
		for muscle in muscles:
			for region in resolve_muscle(muscle):
				scores[region] += weight
	if not scores:
		for region in resolve_body_parts(body_part):
			scores[region] += 0.8
	return dict(scores)


def compute_exercise_roles(row: pd.Series) -> dict[str, str]:
	"""Highest-priority display role per SVG region."""
	region_role: dict[str, str] = {}
	body_part = str(row.get("body_part", ""))
	for csv_col in ROLE_COLUMNS:
		display_role = CSV_TO_ROLE[csv_col]
		muscles = parse_muscle_list(row.get(csv_col, ""))
		for muscle in muscles:
			for region in resolve_muscle(muscle):
				prev = region_role.get(region)
				if prev is None or ROLE_PRIORITY[display_role] < ROLE_PRIORITY[prev]:
					region_role[region] = display_role
	if not region_role:
		for region in resolve_body_parts(body_part):
			region_role.setdefault(region, "Primary")
	return region_role


def compute_plan_scores(exercises_df: pd.DataFrame) -> dict[str, float]:
	"""Aggregate muscle scores across multiple exercises (solver output)."""
	total: dict[str, float] = defaultdict(float)
	for _, row in exercises_df.iterrows():
		for region, score in compute_exercise_scores(row).items():
			total[region] += score
	return dict(total)


# ─── SVG parsing ──────────────────────────────────────────────────


def parse_svg_groups(svg_path: Path) -> dict[str, list[str]]:
	tree = ET.parse(svg_path)
	root = tree.getroot()
	ns_match = re.match(r"\{(.*)\}", root.tag)
	ns = ns_match.group(1) if ns_match else ""

	def tag(name: str) -> str:
		return f"{{{ns}}}{name}" if ns else name

	groups: dict[str, list[str]] = {}
	for g in root.iter(tag("g")):
		gid = g.get("id")
		if gid:
			paths = [
				child.get("d")
				for child in g
				if child.tag == tag("path") and child.get("d")
			]
			if paths:
				groups[gid] = paths
	return groups


# ─── Visualizer ───────────────────────────────────────────────────


class ExerciseBodyVisualizer:
	def __init__(self, df: pd.DataFrame, mode: str = "browse"):
		self.df = df.reset_index(drop=True)
		self.idx = 0
		self.current_url = ""
		self.mode = mode

		self.svg_groups = parse_svg_groups(SVG_FILE)

		self.fig = plt.figure(figsize=(16, 9))

		if mode == "browse":
			gs = self.fig.add_gridspec(
				nrows=3, ncols=2,
				height_ratios=[12, 0.6, 1],
				width_ratios=[3, 1],
			)
		else:
			gs = self.fig.add_gridspec(nrows=1, ncols=2, width_ratios=[3, 1])

		self.ax_body = self.fig.add_subplot(gs[0, 0])
		self.ax_info = self.fig.add_subplot(gs[0, 1])

		self._draw_structure()
		self.muscle_patches: dict[str, list[PathPatch]] = {}
		self._create_muscle_patches()

		self.ax_body.set_xlim(0, 3528.37)
		self.ax_body.set_ylim(3203.47, 0)
		self.ax_body.set_aspect("equal")
		self.ax_body.axis("off")

		self.ax_info.axis("off")

		# Click-to-highlight state
		self._muscle_texts: list[tuple[plt.Text, list[str]]] = []
		self._highlighted_regions: list[str] = []
		self._saved_styles: dict[str, list[tuple]] = {}
		self.fig.canvas.mpl_connect("button_press_event", self._on_click_muscle)

		if mode == "browse":
			# Legend
			self.ax_legend = self.fig.add_subplot(gs[1, :])
			self.ax_legend.axis("off")
			self._draw_legend()

			# Buttons
			btn_gs = gs[2, :].subgridspec(1, 4)
			self.ax_prev = self.fig.add_subplot(btn_gs[0, 0])
			self.ax_next = self.fig.add_subplot(btn_gs[0, 1])
			self.ax_random = self.fig.add_subplot(btn_gs[0, 2])
			self.ax_open = self.fig.add_subplot(btn_gs[0, 3])

			self.btn_prev = Button(self.ax_prev, "Previous")
			self.btn_next = Button(self.ax_next, "Next")
			self.btn_random = Button(self.ax_random, "Random")
			self.btn_open = Button(self.ax_open, "Open description")

			self.btn_prev.on_clicked(self._on_prev)
			self.btn_next.on_clicked(self._on_next)
			self.btn_random.on_clicked(self._on_random)
			self.btn_open.on_clicked(self._on_open_url)
			self.fig.canvas.mpl_connect("key_press_event", self._on_key)

			self._show_exercise()
		else:
			self._show_plan()

	# ── SVG rendering ─────────────────────────────────────────────

	def _draw_structure(self):
		for gid in STRUCTURE_SVG_IDS:
			for d in self.svg_groups.get(gid, []):
				try:
					mpl_path = parse_path(d)
					if gid == "face":
						patch = PathPatch(mpl_path, fc="white", ec="#333", lw=0.8, zorder=2)
					elif "borders" in gid:
						patch = PathPatch(mpl_path, fc="#f0f0f0", ec="#555", lw=1.5, zorder=0)
					else:
						patch = PathPatch(mpl_path, fc="none", ec="#aaa", lw=0.4, zorder=1)
					self.ax_body.add_patch(patch)
				except Exception:
					pass

	def _create_muscle_patches(self):
		for gid in MUSCLE_SVG_IDS:
			self.muscle_patches[gid] = []

		for svg_gid, all_d in self.svg_groups.items():
			if svg_gid in GROUP_SPLITS:
				# Split one SVG group into multiple sub-regions
				for sub_id, indices in GROUP_SPLITS[svg_gid].items():
					for idx in indices:
						if idx < len(all_d):
							try:
								patch = PathPatch(
									parse_path(all_d[idx]),
									fc=INACTIVE_FACE, ec=INACTIVE_EDGE,
									lw=0.5, alpha=INACTIVE_ALPHA, zorder=3,
								)
								self.ax_body.add_patch(patch)
								self.muscle_patches[sub_id].append(patch)
							except Exception:
								pass
			elif svg_gid in MUSCLE_SVG_IDS:
				for d in all_d:
					try:
						patch = PathPatch(
							parse_path(d),
							fc=INACTIVE_FACE, ec=INACTIVE_EDGE,
							lw=0.5, alpha=INACTIVE_ALPHA, zorder=3,
						)
						self.ax_body.add_patch(patch)
						self.muscle_patches[svg_gid].append(patch)
					except Exception:
						pass

	def _draw_legend(self):
		"""Draw a horizontal legend showing role -> color mapping."""
		ax = self.ax_legend
		roles = DISPLAY_ROLES
		n = len(roles)
		total_w = 0.85
		box_w = total_w / n
		y = 0.2
		for i, role in enumerate(roles):
			color, alpha = ROLE_COLORS[role]
			x = 0.05 + i * box_w
			rect = FancyBboxPatch(
				(x, y), box_w * 0.25, 0.55,
				boxstyle="round,pad=0.02", fc=color, alpha=alpha,
				ec=ROLE_EDGE_COLOR, lw=0.8, transform=ax.transAxes,
			)
			ax.add_patch(rect)
			ax.text(
				x + box_w * 0.30, y + 0.25, role,
				fontsize=10, va="center", weight="bold",
				transform=ax.transAxes,
			)

	# ── Coloring ──────────────────────────────────────────────────

	def _color_by_role(self, region_roles: dict[str, str]):
		for gid, patches in self.muscle_patches.items():
			role = region_roles.get(gid)
			for patch in patches:
				if role is None:
					patch.set_facecolor(INACTIVE_FACE)
					patch.set_edgecolor(INACTIVE_EDGE)
					patch.set_alpha(INACTIVE_ALPHA)
				else:
					color, alpha = ROLE_COLORS[role]
					patch.set_facecolor(color)
					patch.set_edgecolor(ROLE_EDGE_COLOR)
					patch.set_alpha(alpha)

	def _color_by_heatmap(self, scores: dict[str, float]):
		max_score = max(scores.values(), default=1.0)
		cmap = plt.colormaps["YlOrRd"]
		for gid, patches in self.muscle_patches.items():
			score = scores.get(gid, 0.0)
			for patch in patches:
				if score <= 0:
					patch.set_facecolor(INACTIVE_FACE)
					patch.set_edgecolor(INACTIVE_EDGE)
					patch.set_alpha(INACTIVE_ALPHA)
				else:
					intensity = min(score / max_score, 1.0)
					patch.set_facecolor(cmap(0.25 + 0.75 * intensity))
					patch.set_edgecolor(ROLE_EDGE_COLOR)
					patch.set_alpha(0.4 + 0.6 * intensity)

	# ── Click-to-highlight ────────────────────────────────────────

	def _on_click_muscle(self, event):
		if event.inaxes != self.ax_info:
			return
		for txt, regions in self._muscle_texts:
			try:
				hit, _ = txt.contains(event)
			except (RuntimeError, AttributeError, TypeError):
				continue
			if hit:
				if self._highlighted_regions == regions:
					self._unhighlight()
				else:
					self._unhighlight()
					self._highlight(regions)
				self.fig.canvas.draw_idle()
				return
		# Clicked info panel but not on a muscle — clear highlight
		if self._highlighted_regions:
			self._unhighlight()
			self.fig.canvas.draw_idle()

	def _highlight(self, regions: list[str]):
		self._highlighted_regions = regions
		for gid in regions:
			saved = []
			for p in self.muscle_patches.get(gid, []):
				alpha = p.get_alpha() or 0.0
				saved.append((p.get_edgecolor(), alpha, p.get_linewidth()))
				p.set_edgecolor("#00e5ff")
				p.set_linewidth(3.5)
				p.set_alpha(max(alpha, 0.85))
				p.set_zorder(10)
			self._saved_styles[gid] = saved

	def _unhighlight(self):
		for gid, saved_list in self._saved_styles.items():
			for p, (ec, alpha, lw) in zip(self.muscle_patches.get(gid, []), saved_list):
				p.set_edgecolor(ec)
				p.set_linewidth(lw)
				p.set_alpha(alpha)
				p.set_zorder(3)
		self._saved_styles.clear()
		self._highlighted_regions = []

	# ── Display ───────────────────────────────────────────────────

	def _show_exercise(self):
		row = self.df.iloc[self.idx]
		roles = compute_exercise_roles(row)
		self._color_by_role(roles)

		name = str(row.get("exercise_name", "?"))
		body_part = str(row.get("body_part", "?"))
		self.current_url = str(row.get("exercise_url", "")).strip()

		self.fig.suptitle(
			f"{self.idx + 1}/{len(self.df)}  \u2022  {name}",
			fontsize=14, weight="bold",
		)

		merged = _merged_muscles(row)

		# Build readable info text — clear state before destroying artists
		self._muscle_texts.clear()
		self._unhighlight()
		self.ax_info.clear()
		self.ax_info.axis("off")

		y = 0.96
		line_h = 0.035

		def put(text, **kw):
			nonlocal y
			defaults = dict(fontsize=10, va="top", transform=self.ax_info.transAxes)
			defaults.update(kw)
			t = self.ax_info.text(0.02, y, text, **defaults)
			y -= line_h
			return t

		put(name, fontsize=12, weight="bold")
		y -= line_h * 0.3
		put(f"Body part:  {body_part}", fontsize=10, color="#555")
		y -= line_h * 0.8

		for role in DISPLAY_ROLES:
			muscles = merged[role]
			if not muscles:
				continue
			color, _ = ROLE_COLORS[role]
			put(f"\u25cf {role}", fontsize=11, weight="bold", color=color)
			for m in muscles:
				regions = resolve_muscle(m)
				t = put(f"    {m}", fontsize=9, color="#333", picker=True)
				if regions:
					self._muscle_texts.append((t, regions))
			y -= line_h * 0.3

		if self.current_url:
			y -= line_h * 0.5
			short = self.current_url if len(self.current_url) < 60 else self.current_url[:57] + "..."
			put(short, fontsize=8, color="#888")

		self.fig.canvas.draw_idle()

	def _show_plan(self):
		scores = compute_plan_scores(self.df)
		self._color_by_heatmap(scores)

		self.fig.suptitle(
			f"Training plan \u2014 {len(self.df)} exercises",
			fontsize=14, weight="bold",
		)

		self.ax_info.clear()
		self.ax_info.axis("off")

		y = 0.96
		line_h = 0.028

		def put(text, **kw):
			nonlocal y
			defaults = dict(fontsize=10, va="top", transform=self.ax_info.transAxes)
			defaults.update(kw)
			self.ax_info.text(0.02, y, text, **defaults)
			y -= line_h

		put("Exercises:", fontsize=11, weight="bold")
		y -= line_h * 0.3
		for _, row in self.df.iterrows():
			put(f"\u2022 {row.get('exercise_name', '?')}", fontsize=9)

		y -= line_h
		put("Muscle engagement:", fontsize=11, weight="bold")
		y -= line_h * 0.3
		max_score = max(scores.values(), default=1.0)
		for region, score in sorted(scores.items(), key=lambda x: -x[1]):
			bar = "\u2588" * int(min(score / max_score * 10, 10))
			put(f"  {region:<16s} {score:5.1f} {bar}", fontsize=9, family="monospace")

		self.fig.canvas.draw_idle()

	# ── Navigation ────────────────────────────────────────────────

	def _on_prev(self, _):
		self.idx = (self.idx - 1) % len(self.df)
		self._show_exercise()

	def _on_next(self, _):
		self.idx = (self.idx + 1) % len(self.df)
		self._show_exercise()

	def _on_random(self, _):
		self.idx = random.randrange(len(self.df))
		self._show_exercise()

	def _on_open_url(self, _):
		if self.current_url:
			webbrowser.open(self.current_url, new=2)

	def _on_key(self, event):
		if event.key in {"right", "d", "n"}:
			self._on_next(event)
		elif event.key in {"left", "a", "p"}:
			self._on_prev(event)
		elif event.key == "r":
			self._on_random(event)
		elif event.key in {"o", "enter"}:
			self._on_open_url(event)


# ─── Data loading ─────────────────────────────────────────────────


def load_data(csv_path: str | None = None, body_part: str | None = None,
              contains: str | None = None) -> pd.DataFrame:
	path = Path(csv_path) if csv_path else DATA_FILE
	if not path.exists():
		raise FileNotFoundError(f"CSV not found: {path}")
	df = pd.read_csv(path)
	if body_part:
		df = df[df["body_part"].str.contains(body_part, case=False, na=False)]
	if contains:
		df = df[df["exercise_name"].str.contains(contains, case=False, na=False)]
	if df.empty:
		raise ValueError("No data after filtering.")
	return df


def load_plan(exercise_names: list[str], csv_path: str | None = None) -> pd.DataFrame:
	"""Load exercises matching a list of names (for solver output)."""
	path = Path(csv_path) if csv_path else DATA_FILE
	df = pd.read_csv(path)
	plan_df = df[df["exercise_name"].isin(exercise_names)]
	if not plan_df.empty:
		return plan_df
	mask = pd.Series(False, index=df.index)
	for name in exercise_names:
		mask |= df["exercise_name"].str.contains(name, case=False, na=False)
	plan_df = df[mask]
	if plan_df.empty:
		raise ValueError(f"No exercises found matching: {exercise_names}")
	return plan_df


# ─── CLI ──────────────────────────────────────────────────────────


def main():
	parser = argparse.ArgumentParser(description="Exercise body visualizer (SVG model).")
	parser.add_argument("--csv", help="Path to CSV file")
	parser.add_argument("--body-part", help="Filter by body_part")
	parser.add_argument("--contains", help="Filter by exercise name")
	parser.add_argument("--plan", nargs="+", help="Plan mode: list of exercise names to aggregate")
	args = parser.parse_args()

	if args.plan:
		plan_df = load_plan(args.plan, args.csv)
		ExerciseBodyVisualizer(plan_df, mode="plan")
	else:
		df = load_data(args.csv, args.body_part, args.contains)
		ExerciseBodyVisualizer(df, mode="browse")

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
