from dataclasses import dataclass


@dataclass
class Sentiment:
    min: float
    avg: float
    max: float
