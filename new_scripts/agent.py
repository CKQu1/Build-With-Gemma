"""
agent.py - Person C's stage.

Takes a Contract A detection event, looks up mission context for the channel,
decides who is affected, builds crew-facing instructions, and then applies the
bounded-autonomy policy from contracts.py.

This deterministic agent is intentionally small and swappable: a Gemma call can
later produce the severity/rationale/action fields, while the policy and output
contract stay unchanged.
"""

from __future__ import annotations

import json

try:  # Allows both `python new_scripts/agent.py` and package-style imports.
    from .contracts import (
        EXAMPLE_DETECTION_EVENT,
        apply_policy,
        get_channel_context,
        make_crew_instruction,
        validate_detection_event,
    )
except ImportError:  # pragma: no cover - exercised when run as a script
    from contracts import (  # type: ignore
        EXAMPLE_DETECTION_EVENT,
        apply_policy,
        get_channel_context,
        make_crew_instruction,
        validate_detection_event,
    )


def _event_duration(event: dict) -> int:
    start, end = event["flagged_range"]
    return int(end) - int(start) + 1


def _overlaps_any(flagged_range: list[int], ranges: list[list[int]]) -> bool:
    start, end = flagged_range
    return any(start <= known_end and end >= known_start for known_start, known_end in ranges)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class MissionAgent:
    """Context-aware anomaly triage for telemetry events."""

    def decide(self, detection_event: dict) -> dict:
        """Convert Contract A into a validated Contract B decision packet."""
        event = validate_detection_event(detection_event)
        context = get_channel_context(event["channel"], event["spacecraft"])

        severity, action = self._recommend_action(event, context)
        confidence = self._confidence(event, context, severity, action)
        rationale = self._rationale(event, context, severity, action)
        instructions = self._crew_instructions(event, context, severity, action)
        next_checks = self._next_checks(event, context, severity)

        reasoning_output = {
            "severity": severity,
            "confidence": confidence,
            "rationale": rationale,
            "recommended_action": action,
        }

        return apply_policy(
            reasoning_output,
            event["event_id"],
            subsystem=context["subsystem"],
            context_summary=context["context_summary"],
            affected_roles=context["affected_roles"],
            crew_instructions=instructions,
            next_checks=next_checks,
        )

    def _recommend_action(self, event: dict, context: dict) -> tuple[str, str]:
        score = float(event["score"])
        duration = _event_duration(event)
        sustained_after = int(context["sustained_after"])
        critical_after = int(context["critical_after"])
        elevated_score = float(context["elevated_score"])
        critical_score = float(context["critical_score"])

        if duration >= critical_after and score >= critical_score:
            return "high", "safe_mode"
        if duration >= sustained_after or score >= critical_score:
            return "high", "escalate"
        if score >= elevated_score:
            return "medium", "isolate_channel"
        return "low", "monitor"

    def _confidence(self, event: dict, context: dict, severity: str, action: str) -> float:
        score = float(event["score"])
        duration = _event_duration(event)
        overlaps_known = _overlaps_any(event["flagged_range"], context["known_anomaly_ranges"])

        confidence = 0.52
        if context["channel"] in {"S-1"}:
            confidence += 0.10
        if score >= float(context["elevated_score"]):
            confidence += 0.08
        if score >= float(context["critical_score"]):
            confidence += 0.08
        if duration >= int(context["sustained_after"]):
            confidence += 0.10
        if overlaps_known:
            confidence += 0.08
        if severity == "high" and action == "safe_mode":
            confidence -= 0.04

        return round(_clamp(confidence, 0.35, 0.95), 2)

    def _rationale(self, event: dict, context: dict, severity: str, action: str) -> str:
        start, end = event["flagged_range"]
        duration = _event_duration(event)
        overlaps_known = _overlaps_any(event["flagged_range"], context["known_anomaly_ranges"])
        known_clause = (
            "It overlaps the known S-1 anomaly window."
            if overlaps_known
            else "It does not overlap any channel-specific anomaly window on file."
        )

        return (
            f"{event['channel']} produced a {event['score']:.2f} {event['method']} score "
            f"over timesteps {start}..{end} ({duration} tick(s)). "
            f"{known_clause} Given the {context['subsystem']} context, this is "
            f"rated {severity} and the recommended action is {action}. "
            f"Protocol note: {context['protocol']}"
        )

    def _crew_instructions(
        self,
        event: dict,
        context: dict,
        severity: str,
        action: str,
    ) -> list[dict]:
        start, end = event["flagged_range"]
        priority = "high" if severity == "high" else "medium" if severity == "medium" else "low"
        watchlist = ", ".join(context["watchlist_channels"]) or "correlated subsystem channels"

        instructions = [
            make_crew_instruction(
                "flight_director",
                priority,
                (
                    f"Assign {event['channel']} ownership to "
                    f"{', '.join(context['owners'])}; keep the event active until "
                    f"timesteps {start}..{end} are explained."
                ),
            ),
            make_crew_instruction(
                context["owners"][0],
                priority,
                (
                    f"Cross-check {event['channel']} against {watchlist}; if "
                    f"the isolation command executes, keep {event['channel']} out "
                    "of automated estimator trust until redundant telemetry agrees."
                ),
            ),
            make_crew_instruction(
                "ground_comms",
                "medium" if severity == "high" else "low",
                (
                    f"Package the raw {event['channel']} slice, command history, "
                    f"and decision packet for the next ground contact window."
                ),
            ),
        ]

        if "systems_engineer" in context["affected_roles"] and context["owners"][0] != "systems_engineer":
            instructions.append(
                make_crew_instruction(
                    "systems_engineer",
                    priority,
                    (
                        "Trend power, thermal, and command-state context around "
                        f"timesteps {start}..{end} to separate sensor failure from "
                        "expected spacecraft mode changes."
                    ),
                )
            )

        if "science_payload_lead" in context["affected_roles"]:
            instructions.append(
                make_crew_instruction(
                    "science_payload_lead",
                    priority,
                    (
                        f"Flag science products generated during the {event['channel']} "
                        "interval for pointing-quality review before release."
                    ),
                )
            )

        if action in {"safe_mode", "escalate"}:
            instructions.append(
                make_crew_instruction(
                    "flight_director",
                    "high",
                    (
                        "Hold autonomous escalation for ground authorization unless "
                        "corroborating channels show broader spacecraft instability."
                    ),
                )
            )

        return instructions

    def _next_checks(self, event: dict, context: dict, severity: str) -> list[str]:
        start, end = event["flagged_range"]
        checks = [
            f"Compare {event['channel']} with {', '.join(context['watchlist_channels']) or 'peer channels'}.",
            f"Inspect command history and spacecraft mode around timesteps {start}..{end}.",
            f"Track whether the flagged range exceeds {context['sustained_after']} timesteps.",
        ]

        if context["known_anomaly_ranges"]:
            checks.append(
                f"Check against known {event['channel']} anomaly ranges: "
                f"{context['known_anomaly_ranges']}."
            )
        if severity == "high":
            checks.append("Prepare a ground-review packet before any safe-mode action.")

        return checks


if __name__ == "__main__":
    decision = MissionAgent().decide(EXAMPLE_DETECTION_EVENT)
    print(json.dumps(decision, indent=2))
