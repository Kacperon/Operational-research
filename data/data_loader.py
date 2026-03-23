from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from data.abstract_data_loader import AbstractDataLoader
from planner.exercise import Exercise


class DataLoader(AbstractDataLoader):
	"""Load exercise definitions from the ExRx CSV dataset."""

	EMPTY_VALUES = {"", "nan", "none"}
	NO_SIGNIFICANT_MARKER = "no significant stabilizers"
	REQUIRED_COLUMNS = {"exercise_name", "Target", "Synergists", "Stabilizers"}

	def __init__(self, csv_path: str | Path) -> None:
		"""
		Initialize the loader.

		Parameters:
			csv_path (str | Path): Path to a CSV file that contains at least
				``exercise_name``, ``exercise_url``, ``body_part``, ``Target``,
				``Synergists``, and ``Stabilizers`` columns.
		"""
		self.csv_path = Path(csv_path).expanduser().resolve()
		self._raw_df: pd.DataFrame | None = None
		self._muscle_group_count: int | None = None

	@staticmethod
	def _normalize_text(value: str) -> str:
		"""Normalize whitespace and casing so muscle names map consistently."""
		compact = re.sub(r"\s+", " ", value.strip())
		return compact.lower()

	@classmethod
	def _parse_muscle_list(cls, value: str | None) -> list[str]:
		"""
		Parse a semicolon-separated muscle list from a CSV cell.

		Empty values and entries like "No significant stabilizers" are ignored.
		"""
		if value is None:
			return []

		text = str(value).strip()
		if cls._normalize_text(text) in cls.EMPTY_VALUES:
			return []

		if cls.NO_SIGNIFICANT_MARKER in cls._normalize_text(text):
			return []

		parts = [part.strip() for part in text.split(";")]
		return [part for part in parts if part]

	def _read_csv(self) -> pd.DataFrame:
		"""Read and cache the source CSV as a pandas DataFrame."""
		if self._raw_df is None:
			if not self.csv_path.exists():
				raise FileNotFoundError(f"CSV file does not exist: {self.csv_path}")

			df = pd.read_csv(self.csv_path)
			missing = self.REQUIRED_COLUMNS - set(df.columns)
			if missing:
				missing_str = ", ".join(sorted(missing))
				raise ValueError(f"Missing required CSV columns: {missing_str}")

			self._raw_df = df

		return self._raw_df

	def raw(self) -> pd.DataFrame:
		"""Return the raw dataset loaded from CSV."""
		return self._read_csv().copy()

	def exercises(self) -> np.ndarray[Exercise]:
		"""Return parsed exercises as an array of Exercise objects."""
		df = self._read_csv()

		exercises: list[Exercise] = []
		for row in df.itertuples(index=False):
			targets = self._parse_muscle_list(getattr(row, "Target", None))
			synergists = self._parse_muscle_list(getattr(row, "Synergists", None))
			stabilizers = self._parse_muscle_list(getattr(row, "Stabilizers", None))

			exercises.append(
				Exercise(
					name=str(getattr(row, "exercise_name", "")).strip(),
					targets=targets,
					synergists=synergists,
					stabilizers=stabilizers,
				)
			)

		return np.array(exercises, dtype=object)

	def muscle_group_count(self) -> int:
		"""Return the count of unique muscles across target/synergist/stabilizer roles."""
		if self._muscle_group_count is None:
			df = self._read_csv()
			muscles: set[str] = set()

			for row in df.itertuples(index=False):
				for value in (
					getattr(row, "Target", None),
					getattr(row, "Synergists", None),
					getattr(row, "Stabilizers", None),
				):
					for muscle in self._parse_muscle_list(value):
						muscles.add(self._normalize_text(muscle))

			self._muscle_group_count = len(muscles)

		return self._muscle_group_count
