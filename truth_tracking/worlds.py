"""
worlds.py

Defines the explicit representation of worlds for n propositional variables.
A world is represented as an integer in [0, 2^n - 1], corresponding to an n-bit assignment.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WorldSpace:
    """A space of 2^n explicitly represented worlds."""

    n: int

    @property
    def size(self) -> int:
        return 2 ** self.n

    def all_worlds(self) -> np.ndarray:
        """Return all worlds as an array of integers [0..2^n - 1]."""
        return np.arange(self.size, dtype=np.int64)

    def sample_true_world(self, rng: np.random.Generator) -> int:
        """Select a true world uniformly at random."""
        return int(rng.integers(0, self.size))


def world_to_bits(world: int, n: int) -> str:
    """Return the n-bit representation of a world index."""
    if world < 0:
        raise ValueError("world must be non-negative.")
    return format(world, f"0{n}b")
