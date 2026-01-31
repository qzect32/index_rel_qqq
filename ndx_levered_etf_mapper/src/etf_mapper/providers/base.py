from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import pandas as pd

@dataclass(frozen=True)
class ProviderResult:
    etfs: pd.DataFrame
    edges: pd.DataFrame

class Provider(Protocol):
    name: str
    def fetch(self) -> ProviderResult: ...
