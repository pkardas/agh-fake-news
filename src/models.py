from dataclasses import dataclass


@dataclass
class News:
    is_fake: bool
    body: str

    @property
    def is_true(self) -> bool:
        return not self.is_fake
