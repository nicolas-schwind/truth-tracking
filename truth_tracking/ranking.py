"""
ranking.py

Implements ranking functions and improvement operator f_x.

Ranking function is stored as a vector of nonnegative integers, one per world.
Beliefs are the set of worlds with rank 0.
"""

from dataclasses import dataclass
from typing import Set
import numpy as np


@dataclass
class RankingFunction:
    """
    Ranking function rf: worlds -> non-negative integers.
    Represented by numpy array ranks[w].
    """

    ranks: np.ndarray

    @classmethod
    def flat(cls, num_worlds: int) -> "RankingFunction":
        """Ranking function mapping all worlds to 0."""
        return cls(ranks=np.zeros(num_worlds, dtype=np.int64))

    def normalize(self) -> None:
        """Normalize so that minimum rank is 0."""
        m = int(self.ranks.min())
        if m != 0:
            self.ranks -= m

    def belief(self) -> Set[int]:
        """Bel(rf) = set of worlds mapped to 0."""
        zeros = np.flatnonzero(self.ranks == 0)
        return set(map(int, zeros))

    def improve(self, formula: np.ndarray, x: int) -> None:
        """
        Apply improvement operator f_x for a formula F (given as an array of worlds).
        For w in F: rf(w) := rf(w) - x
        then normalize.
        """
        if x <= 0:
            raise ValueError("x must be a strictly positive integer.")
        self.ranks[formula] -= x
        self.normalize()
