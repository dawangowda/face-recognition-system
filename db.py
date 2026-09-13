"""
Enrollment store and the matching logic.

Design decision worth defending in the interview: this is a flat numpy matrix,
not a vector database. At enrollment scales of tens to thousands of identities,
a brute-force dot product against the full gallery is exact, sub-millisecond,
and has no operational cost. An ANN index (FAISS, hnswlib) trades exactness for
speed and only starts paying for itself somewhere north of ~100k vectors.
Reaching for one here would be over-engineering.

Each identity is stored as the mean of its enrolled embeddings (a centroid),
re-normalised to unit length. Averaging several shots of one person suppresses
per-image noise from lighting and pose, and keeps matching O(identities)
instead of O(images).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Match:
    name: str | None  # None means rejected as unknown
    score: float  # cosine similarity to the best gallery entry
    runner_up: str | None = None
    runner_up_score: float = 0.0

    @property
    def is_known(self) -> bool:
        return self.name is not None

    @property
    def margin(self) -> float:
        """Gap to the second-best identity. A thin margin means low confidence."""
        return self.score - self.runner_up_score


class FaceDB:
    def __init__(self, path: str | Path = "gallery.npz") -> None:
        self.path = Path(path)
        # name -> list of individual embeddings (kept so re-enrolling can
        # extend an identity rather than overwrite it)
        self._samples: dict[str, list[np.ndarray]] = {}
        self._names: list[str] = []
        self._matrix: np.ndarray | None = None  # (n_identities, dim)
        if self.path.exists():
            self.load()

    # ---------- persistence ----------

    def save(self) -> None:
        flat, owners = [], []
        for name, vecs in self._samples.items():
            for v in vecs:
                flat.append(v)
                owners.append(name)
        arr = np.stack(flat) if flat else np.zeros((0, 1), dtype=np.float32)
        np.savez_compressed(
            self.path, embeddings=arr, owners=np.array(owners, dtype=object)
        )

    def load(self) -> None:
        data = np.load(self.path, allow_pickle=True)
        self._samples = {}
        for vec, name in zip(data["embeddings"], data["owners"]):
            self._samples.setdefault(str(name), []).append(
                np.asarray(vec, dtype=np.float32)
            )
        self._rebuild()

    # ---------- enrollment ----------

    def enroll(self, name: str, embeddings: list[np.ndarray]) -> int:
        """Add embeddings for `name`. Returns that identity's total sample count."""
        if not embeddings:
            raise ValueError("no embeddings supplied")
        self._samples.setdefault(name, []).extend(
            np.asarray(e, dtype=np.float32).ravel() for e in embeddings
        )
        self._rebuild()
        return len(self._samples[name])

    def remove(self, name: str) -> bool:
        if name not in self._samples:
            return False
        del self._samples[name]
        self._rebuild()
        return True

    def _rebuild(self) -> None:
        """Recompute the centroid matrix. Cheap; called after every write."""
        self._names = sorted(self._samples)
        if not self._names:
            self._matrix = None
            return
        rows = []
        for name in self._names:
            c = np.mean(np.stack(self._samples[name]), axis=0)
            n = np.linalg.norm(c)
            rows.append(c / n if n > 1e-10 else c)
        self._matrix = np.stack(rows).astype(np.float32)

    # ---------- matching ----------

    def identify(self, embedding: np.ndarray, threshold: float) -> Match:
        """
        Compare one probe embedding against every enrolled centroid.

        Because all vectors are unit length, the dot product *is* the cosine
        similarity. Anything scoring below `threshold` is rejected as unknown -
        this is the open-set rejection mechanism, and it is the reason the
        system can say "I don't know" instead of always naming its closest guess.
        """
        if self._matrix is None:
            return Match(name=None, score=0.0)

        probe = np.asarray(embedding, dtype=np.float32).ravel()
        n = np.linalg.norm(probe)
        if n > 1e-10:
            probe = probe / n

        scores = self._matrix @ probe
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        best_score = float(scores[best])

        runner_up, runner_up_score = None, 0.0
        if len(order) > 1:
            second = int(order[1])
            runner_up = self._names[second]
            runner_up_score = float(scores[second])

        return Match(
            name=self._names[best] if best_score >= threshold else None,
            score=best_score,
            runner_up=runner_up,
            runner_up_score=runner_up_score,
        )

    # ---------- introspection ----------

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def stats(self) -> dict:
        return {
            "identities": len(self._names),
            "total_samples": sum(len(v) for v in self._samples.values()),
            "per_identity": {k: len(v) for k, v in sorted(self._samples.items())},
        }


def load_threshold(path: str | Path = "threshold.json", fallback: float = 0.4) -> float:
    """
    Read the calibrated threshold produced by evaluate.py.

    Falls back to the backend default if evaluation has not been run yet.
    """
    p = Path(path)
    if p.exists():
        try:
            return float(json.loads(p.read_text())["threshold"])
        except Exception:
            pass
    return fallback
