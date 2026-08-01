"""
streamer.py — Person A's stage. Turns a static .npy telemetry file into a
paced, tick-by-tick "live" feed.

Hands off to Person B (detector): a stream of {"t": int, "value": float} dicts.
No contracts.py import needed here — this stage produces raw ticks, not
Contract A/B dicts.

RUN `python streamer.py` for a quick smoke test (prints a few ticks fast,
around the S-1 anomaly window).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

# Resolve relative to this file, not the caller's cwd, so `stream()` works
# the same whether you run it from the project root or import it elsewhere.
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "nasa" / "test"


def stream(channel: str = "S-1", start: int = 5200, speed: float = 0.01):
    """Yield {"t", "value"} dicts one at a time, pacing with time.sleep(speed).

    channel: which <channel>.npy file to read from data/test/.
    start:   timestep index to jump to (skip the boring normal stretch).
    speed:   seconds to sleep between ticks. Use 0 for a fast smoke test.
    """
    path = DATA_DIR / f"{channel}.npy"
    assert path.exists(), f"[streamer] no such channel file: {path}"

    signal = np.load(path)[:, 0]
    assert 0 <= start < len(signal), (
        f"[streamer] start={start} out of range for {channel} "
        f"(0..{len(signal) - 1})"
    )

    for t in range(start, len(signal)):
        yield {"t": t, "value": float(signal[t])}
        if speed:
            time.sleep(speed)


if __name__ == "__main__":
    count = 0
    for tick in stream(channel="S-1", start=5200, speed=0):
        print(tick)
        count += 1
        if count >= 10:
            break
    print(f"... streamed {count} ticks OK, first tick at t={5200}")
