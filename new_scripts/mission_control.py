"""
mission_control.py - end-to-end demo for the new real-time response loop.

Run from the repo root:

    python new_scripts/mission_control.py

It streams S-1 telemetry, emits the first anomaly detection, and immediately
prints the context-aware crew decision packet.
"""

from __future__ import annotations

import json

try:  # Allows both `python new_scripts/mission_control.py` and package imports.
    from .agent import MissionAgent
    from .detector import Detector
    from .streamer import stream
except ImportError:  # pragma: no cover - exercised when run as a script
    from agent import MissionAgent  # type: ignore
    from detector import Detector  # type: ignore
    from streamer import stream  # type: ignore


def run_once(channel: str = "S-1", spacecraft: str = "SMAP", start: int = 5200) -> dict | None:
    """Return the first decision packet produced by the live response loop."""
    detector = Detector(channel=channel, spacecraft=spacecraft)
    agent = MissionAgent()

    for tick in stream(channel=channel, start=start, speed=0):
        event = detector.detect(tick["t"], tick["value"])
        if event:
            return agent.decide(event)

    return None


if __name__ == "__main__":
    decision = run_once()
    if decision is None:
        print("No anomaly fired.")
    else:
        print(json.dumps(decision, indent=2))
