"""Shared calibration standards. File values are Å⁻¹; fitting uses nm⁻¹."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class Calibrant:
    name: str
    labels: tuple[str, ...]
    q_nm: tuple[float, ...]

    def q(self, peak: int) -> float:
        """q in nm⁻¹ for a one-based peak index (harmonic order for lamellae)."""
        if not 1 <= peak <= len(self.q_nm):
            raise ValueError(f"{self.name}: unknown peak index {peak}")
        return self.q_nm[peak - 1]

    def options(self) -> dict[str, int]:
        return {f"{label} · {q / 10:.5g} Å⁻¹": i
                for i, (label, q) in enumerate(zip(self.labels, self.q_nm), 1)}

    def nearest(self, q_nm: float) -> int:
        return min(range(1, len(self.q_nm) + 1), key=lambda i: abs(self.q(i) - q_nm))


def _positive_q(value):
    q = float(value)
    if not math.isfinite(q) or q <= 0:
        raise ValueError("Calibrant q values must be finite and positive")
    return q * 10.0


def load_calibrants(extra_path=None) -> dict[str, Calibrant]:
    """Load bundled standards plus optional user definitions, keyed by stable ID.

    ``SMI_BROWSER_CALIBRANTS_FILE`` points to an additional JSON list. Matching
    IDs override bundled entries. Both files use the same schema. Invalid
    definitions fail explicitly rather than silently calibrating against a typo.
    """
    paths = [Path(__file__).with_name("calibrants.json")]
    extra_path = extra_path or os.environ.get("SMI_BROWSER_CALIBRANTS_FILE")
    if extra_path:
        paths.append(Path(extra_path).expanduser())
    result = {}
    for path in paths:
        entries = json.loads(path.read_text())
        if not isinstance(entries, list):
            raise ValueError(f"{path}: expected a JSON list of calibrants")
        seen = set()
        for entry in entries:
            key, name = entry["id"], entry["name"]
            if not isinstance(key, str) or not key.strip() or key in seen:
                raise ValueError(f"{path}: calibrant IDs must be nonempty and unique")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{path}: calibrant name must be nonempty")
            seen.add(key)
            if ("q1_angstrom_inv" in entry) == ("peaks" in entry):
                raise ValueError(f"{key}: specify either q1_angstrom_inv or peaks")
            if "q1_angstrom_inv" in entry:
                q1 = _positive_q(entry["q1_angstrom_inv"])
                maximum = entry.get("max_order", 9)
                if type(maximum) is not int or maximum < 1:
                    raise ValueError(f"{key}: max_order must be a positive integer")
                labels = tuple(f"Order {i}" for i in range(1, maximum + 1))
                qs = tuple(q1 * i for i in range(1, maximum + 1))
            else:
                peaks = entry["peaks"]
                labels = tuple(p["label"] for p in peaks)
                qs = tuple(_positive_q(p["q_angstrom_inv"]) for p in peaks)
            if not qs or any(not isinstance(s, str) or not s.strip() for s in labels) or len(set(labels)) != len(labels):
                raise ValueError(f"{key}: provide at least one peak with unique, nonempty labels")
            if any(b <= a for a, b in zip(qs, qs[1:])):
                raise ValueError(f"{key}: peaks must have strictly increasing q values")
            result[key] = Calibrant(name, labels, qs)
    if len({c.name for c in result.values()}) != len(result):
        raise ValueError("Calibrant display names must be unique")
    return result
