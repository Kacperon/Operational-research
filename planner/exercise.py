from pydantic import BaseModel
from typing import Sequence


class Exercise(BaseModel):
    name: str
    targets: Sequence[str]
    synergists: Sequence[str]
    stabilizers: Sequence[str]
