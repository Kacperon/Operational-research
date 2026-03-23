from dataclasses import dataclass
from typing import Sequence

@dataclass
class Exercise:
    targets: Sequence[int]
    synergists: Sequence[int]
    stabilizers: Sequence[int]
