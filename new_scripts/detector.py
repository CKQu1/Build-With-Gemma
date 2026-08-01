"""
detector.py — Person B's stage. Watches one telemetry value at a time and
flags anomalies as Contract A events for the Gemma agent (Person C).

Emits ONLY when flagged — normal ticks return None. Never hand-builds the
Contract A dict; always goes through make_detection_event() from
contracts.py so a typo can't slip a bad key downstream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # Allows both `python new_scripts/detector.py` and package-style imports.
    from .contracts import make_detection_event
except ImportError:  # pragma: no cover - exercised when run as a script
    from contracts import make_detection_event  # type: ignore


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "nasa" / "test"


class Detector:
    """Rolling z-score anomaly detector.

    window: how many recent ticks to keep for the rolling mean/std baseline.
    thresh: z-score above which a tick is flagged.
    """

    def __init__(
        self,
        window: int = 200,
        thresh: float = 5.0,
        channel: str = "S-1",
        spacecraft: str = "SMAP",
    ):
        self.buf: list[float] = []
        self.window = window
        self.thresh = thresh
        self.channel = channel
        self.spacecraft = spacecraft
        self.active_start: int | None = None
        self.peak_score = 0.0

    def detect(self, t: int, value: float):
        """One tick in. Returns a Contract A dict if flagged, else None."""
        z = 0.0
        if len(self.buf) >= self.window:
            m, s = np.mean(self.buf), np.std(self.buf)
            z = abs(value - m) / (s + 1e-6)

        self.buf.append(value)
        if len(self.buf) > self.window:
            self.buf.pop(0)

        if z > self.thresh:
            if self.active_start is None:
                self.active_start = t
                self.peak_score = z
            else:
                self.peak_score = max(self.peak_score, z)
            return make_detection_event(
                self.channel,
                self.spacecraft,
                t,
                [self.active_start, t],
                self.peak_score,
                "zscore",
            )

        self.active_start = None
        self.peak_score = 0.0
        return None


if __name__ == "__main__":
    signal = np.load(DATA_DIR / "S-1.npy")[:, 0]
    det = Detector(channel="S-1", spacecraft="SMAP")

    fired = []
    for t in range(len(signal)):
        ev = det.detect(t, signal[t])
        if ev:
            fired.append(ev)
            print(ev)

    if fired:
        print(f"\n{len(fired)} events fired, "
              f"t={fired[0]['timestep']}..{fired[-1]['timestep']}")
    else:
        print("\nno events fired")
