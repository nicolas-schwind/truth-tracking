"""
formula_generation.py

Random generation of formulas under the (p,q) protocol:
- true world w* included with prob p
- any other world included with prob q < p
- reject empty formulas (retry)

Supports both fixed-q and per-world random-q modes.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FormulaGenerator:
    """Generates random formulas Fi according to the threshold protocol."""

    p: float
    q: float | None
    true_world: int
    num_worlds: int
    q_mode: str = "uniform"
    max_retries: int = 10000
    marked_worlds: np.ndarray | None = None

    def generate(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a non-empty formula as a numpy array of world indices."""
        if not (0.0 < self.p < 1.0):
            raise ValueError("Must have 0 < p < 1.")
        if self.q_mode not in {"uniform", "fixed"}:
            raise ValueError("q_mode must be 'uniform' or 'fixed'.")
        if self.q_mode == "fixed" or self.marked_worlds is not None:
            if self.q is None or not (0.0 < self.q < self.p):
                raise ValueError("Must have 0 < q < p when q_mode='fixed' or using marked worlds.")
        if self.marked_worlds is not None:
            if np.any(self.marked_worlds == self.true_world):
                raise ValueError("marked_worlds must exclude the true world.")

        for _ in range(self.max_retries):
            # Probability vector: p for true_world, q for others.
            if self.marked_worlds is not None:
                probs = np.zeros(self.num_worlds, dtype=np.float64)
                probs[self.marked_worlds] = float(self.q)
                probs[self.true_world] = self.p
            else:
                if self.q_mode == "fixed":
                    probs = np.full(self.num_worlds, float(self.q), dtype=np.float64)
                else:
                    probs = rng.random(self.num_worlds) * self.p
                probs[self.true_world] = self.p

            included = rng.random(self.num_worlds) < probs
            worlds = np.flatnonzero(included)

            if len(worlds) > 0:
                return worlds.astype(np.int64)

        raise RuntimeError("Failed to generate a non-empty formula within max_retries.")
