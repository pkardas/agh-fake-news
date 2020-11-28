from dataclasses import dataclass
from typing import Tuple


@dataclass
class Sentiment:
    min: float
    avg: float
    max: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return self.min, self.avg, self.max
