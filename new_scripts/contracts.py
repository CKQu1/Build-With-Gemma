"""
contracts.py - the single source of truth for the real-time response pipeline.

ALL teammates import from this file. It defines the exact shape of the two
dicts that cross stage boundaries, plus the shared mission context used to turn
a raw anomaly into an operational decision.

    Person A (streamer) -> Person B (detector) -> Person C (agent) -> Person D (display)
                                      |                    |
                                 CONTRACT A           CONTRACT B

HOW TO USE THIS FILE
--------------------
- If your upstream stage is not ready, build against the EXAMPLE_* dict below.
  It has the exact same shape as the real thing, so swapping later changes
  nothing.
- Right before you EMIT a dict, call the matching validate_*() function. It
  fails loudly on a typo NOW instead of silently at hour 4.
- Do not add or rename keys without telling the whole team. This file is the
  agreement. If it changes, everyone needs to know.

RUN `python contracts.py` TO SELF-TEST.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional


# ---------------------------------------------------------------------------
# Allowed value sets (single source of truth for the string enums)
# ---------------------------------------------------------------------------

SEVERITIES = {"low", "medium", "high"}
PRIORITIES = {"low", "medium", "high"}
ACTIONS = {"monitor", "isolate_channel", "safe_mode", "escalate"}
METHODS = {"zscore", "lstm", "isoforest"}
POLICY_DECISIONS = {"AUTONOMOUS_ACT", "HOLD_LOW_CONFIDENCE", "QUEUE_FOR_GROUND"}

CREW_ROLES = {
    "flight_director",
    "adcs_officer",
    "systems_engineer",
    "science_payload_lead",
    "ground_comms",
}

ROLE_LABELS = {
    "flight_director": "Flight Director",
    "adcs_officer": "ADCS Officer",
    "systems_engineer": "Spacecraft Systems Engineer",
    "science_payload_lead": "Science Payload Lead",
    "ground_comms": "Ground Communications",
}

# Which actions the spacecraft is pre-cleared to take on its own.
# The policy engine (Person C) uses this. Kept here so it is visible to
# everyone.
AUTONOMOUS_OK = {"monitor", "isolate_channel"}
NEEDS_GROUND = {"safe_mode", "escalate"}
CONFIDENCE_GATE = 0.60  # below this, hold for human review


# ===========================================================================
# CONTRACT A - Detector (Person B) -> Agent (Person C)
# Emitted ONLY when the detector flags an anomaly. Otherwise the detector
# returns None.
# ===========================================================================

EXAMPLE_DETECTION_EVENT = {
    "event_id": "evt_S1_5305",      # unique id, convention: evt_<channel>_<timestep>
    "channel": "S-1",               # telemetry channel id from the CSV
    "spacecraft": "SMAP",           # "SMAP" or "MSL"
    "timestep": 5305,               # int, index where the flag fired
    "flagged_range": [5300, 5741],   # [start, end] inclusive index range
    "score": 7.8,                   # float, z-score peak OR lstm error peak
    "method": "zscore",             # "zscore" | "lstm" | "isoforest"
    "channel_prefix": "S",          # first letter/group used to look up context
}

DETECTION_EVENT_KEYS = set(EXAMPLE_DETECTION_EVENT.keys())


def validate_detection_event(d: dict) -> dict:
    """Person B calls this right before emitting a detection event."""
    missing = DETECTION_EVENT_KEYS - d.keys()
    extra = d.keys() - DETECTION_EVENT_KEYS
    assert not missing, f"[Contract A] detection event MISSING keys: {missing}"
    assert not extra, f"[Contract A] detection event has UNEXPECTED keys: {extra}"

    assert isinstance(d["event_id"], str), "[Contract A] event_id must be str"
    assert isinstance(d["channel"], str), "[Contract A] channel must be str"
    assert d["spacecraft"] in {"SMAP", "MSL"}, (
        f"[Contract A] spacecraft must be SMAP or MSL, got {d['spacecraft']!r}"
    )
    assert isinstance(d["timestep"], int), "[Contract A] timestep must be int"

    rng = d["flagged_range"]
    assert (
        isinstance(rng, (list, tuple))
        and len(rng) == 2
        and all(isinstance(x, int) for x in rng)
        and rng[0] <= rng[1]
    ), f"[Contract A] flagged_range must be [start, end] ints with start<=end, got {rng!r}"

    assert isinstance(d["score"], (int, float)), "[Contract A] score must be a number"
    assert d["method"] in METHODS, (
        f"[Contract A] method must be one of {METHODS}, got {d['method']!r}"
    )
    assert isinstance(d["channel_prefix"], str) and len(d["channel_prefix"]) >= 1, (
        "[Contract A] channel_prefix must be a non-empty str"
    )
    return d


def make_detection_event(
    channel: str,
    spacecraft: str,
    timestep: int,
    flagged_range,
    score: float,
    method: str,
) -> dict:
    """Helper so Person B does not hand-build the dict."""
    event = {
        "event_id": f"evt_{channel.replace('-', '')}_{timestep}",
        "channel": channel,
        "spacecraft": spacecraft,
        "timestep": int(timestep),
        "flagged_range": [int(flagged_range[0]), int(flagged_range[1])],
        "score": float(score),
        "method": method,
        "channel_prefix": channel.split("-")[0],
    }
    return validate_detection_event(event)


# ===========================================================================
# Shared mission context - what a channel means operationally.
#
# The public NASA data anonymizes exact channel names. The first letter still
# gives the channel family, so exact channel entries can layer mission-team
# assumptions on top of prefix-level protocol. For S-1 we keep the uncertainty
# explicit while still giving the agent enough context to route instructions.
# ===========================================================================

CHANNEL_CONTEXTS = {
    "S-1": {
        "spacecraft": "SMAP",
        "subsystem": "sensor/attitude telemetry",
        "context_summary": (
            "S-1 is an anonymized SMAP sensor-group stream. In this framework it "
            "is treated as an attitude/state-estimation quality signal that can "
            "affect pointing confidence, autonomous navigation quality, and "
            "science data trust."
        ),
        "nominal_behavior": (
            "Values are pre-scaled between -1 and 1. A brief high-sigma point can "
            "be a transient sensor glitch; a sustained excursion can mean the "
            "estimator is ingesting a degraded sensor."
        ),
        "known_anomaly_ranges": [[5300, 5747]],
        "known_anomaly_class": "point",
        "affected_roles": [
            "flight_director",
            "adcs_officer",
            "systems_engineer",
            "science_payload_lead",
            "ground_comms",
        ],
        "owners": ["adcs_officer", "systems_engineer"],
        "watchlist_channels": ["S-2", "A-*", "D-*"],
        "sustained_after": 50,
        "critical_after": 200,
        "elevated_score": 5.0,
        "critical_score": 9.0,
        "protocol": (
            "For S-1, isolate the channel from automated trust if a sharp spike "
            "exceeds 5 sigma. If the anomaly persists beyond 50 timesteps, treat "
            "it as a possible attitude/state-estimation fault, cross-check "
            "redundant sensor channels, and queue ground review. Avoid safe mode "
            "unless corroborating channels or command history indicate broader "
            "spacecraft instability."
        ),
    }
}

PREFIX_CONTEXTS = {
    "S": {
        "spacecraft": None,
        "subsystem": "sensor/attitude telemetry",
        "context_summary": (
            "Anonymized S-prefix channels are handled as spacecraft sensor or "
            "attitude-quality streams."
        ),
        "nominal_behavior": (
            "Sharp single-point deviations are often transient; sustained "
            "deviations can degrade navigation or pointing confidence."
        ),
        "known_anomaly_ranges": [],
        "known_anomaly_class": "unknown",
        "affected_roles": [
            "flight_director",
            "adcs_officer",
            "systems_engineer",
            "ground_comms",
        ],
        "owners": ["adcs_officer"],
        "watchlist_channels": ["S-*", "A-*", "D-*"],
        "sustained_after": 50,
        "critical_after": 200,
        "elevated_score": 5.0,
        "critical_score": 9.0,
        "protocol": (
            "Channel prefix S: sharp single-point deviations above 5 sigma are "
            "usually transient sensor glitches unless sustained across more than "
            "50 timesteps, in which case escalate."
        ),
    },
    "P": {
        "spacecraft": None,
        "subsystem": "power telemetry",
        "context_summary": "P-prefix channels are treated as power subsystem streams.",
        "nominal_behavior": (
            "Power deviations may correlate with eclipse entry or exit and "
            "expected solar-array load changes."
        ),
        "known_anomaly_ranges": [],
        "known_anomaly_class": "unknown",
        "affected_roles": ["flight_director", "systems_engineer", "ground_comms"],
        "owners": ["systems_engineer"],
        "watchlist_channels": ["P-*", "T-*"],
        "sustained_after": 25,
        "critical_after": 100,
        "elevated_score": 4.5,
        "critical_score": 8.0,
        "protocol": (
            "Channel prefix P: sustained drops with no orbital or load-change "
            "explanation may indicate a real power fault."
        ),
    },
    "T": {
        "spacecraft": None,
        "subsystem": "thermal telemetry",
        "context_summary": "T-prefix channels are treated as thermal subsystem streams.",
        "nominal_behavior": (
            "Gradual thermal drift can be normal across an orbit; abrupt jumps "
            "may indicate heater or radiator behavior."
        ),
        "known_anomaly_ranges": [],
        "known_anomaly_class": "unknown",
        "affected_roles": ["flight_director", "systems_engineer", "ground_comms"],
        "owners": ["systems_engineer"],
        "watchlist_channels": ["T-*", "P-*"],
        "sustained_after": 100,
        "critical_after": 300,
        "elevated_score": 4.5,
        "critical_score": 8.0,
        "protocol": (
            "Channel prefix T: gradual drifts are normal across an orbit; abrupt "
            "jumps may indicate a heater or radiator issue."
        ),
    },
    "R": {
        "spacecraft": None,
        "subsystem": "radiation environment telemetry",
        "context_summary": "R-prefix channels are treated as radiation/environment streams.",
        "nominal_behavior": (
            "Radiation spikes can be environmental; correlate with orbital "
            "location before treating them as hardware faults."
        ),
        "known_anomaly_ranges": [],
        "known_anomaly_class": "unknown",
        "affected_roles": ["flight_director", "systems_engineer", "ground_comms"],
        "owners": ["systems_engineer"],
        "watchlist_channels": ["R-*", "E-*"],
        "sustained_after": 30,
        "critical_after": 120,
        "elevated_score": 5.0,
        "critical_score": 9.0,
        "protocol": (
            "Channel prefix R: spikes over known radiation regions may be "
            "environmental effects, not spacecraft faults."
        ),
    },
    "E": {
        "spacecraft": None,
        "subsystem": "engineering telemetry",
        "context_summary": "E-prefix channels are treated as general engineering streams.",
        "nominal_behavior": (
            "Evaluate magnitude, duration, and whether commands were active in "
            "the same window."
        ),
        "known_anomaly_ranges": [],
        "known_anomaly_class": "unknown",
        "affected_roles": ["flight_director", "systems_engineer", "ground_comms"],
        "owners": ["systems_engineer"],
        "watchlist_channels": ["E-*"],
        "sustained_after": 50,
        "critical_after": 200,
        "elevated_score": 5.0,
        "critical_score": 9.0,
        "protocol": (
            "Channel prefix E: short single-point spikes are typically transient; "
            "sustained deviations warrant escalation."
        ),
    },
}

DEFAULT_CONTEXT = {
    "spacecraft": None,
    "subsystem": "unknown telemetry",
    "context_summary": (
        "No exact channel context is on file. The agent should route this to the "
        "flight director and owning subsystem engineer for review."
    ),
    "nominal_behavior": (
        "Assess magnitude and duration. Short single-point spikes are usually "
        "transient; sustained deviations warrant escalation."
    ),
    "known_anomaly_ranges": [],
    "known_anomaly_class": "unknown",
    "affected_roles": ["flight_director", "systems_engineer", "ground_comms"],
    "owners": ["systems_engineer"],
    "watchlist_channels": [],
    "sustained_after": 50,
    "critical_after": 200,
    "elevated_score": 5.0,
    "critical_score": 9.0,
    "protocol": (
        "No specific protocol on file for this channel group. Assess deviation "
        "magnitude and duration; short single-point spikes are usually transient, "
        "sustained deviations warrant escalation."
    ),
}


def get_channel_context(channel: str, spacecraft: Optional[str] = None) -> dict:
    """Return exact channel context, prefix context, or a conservative default."""
    if channel in CHANNEL_CONTEXTS:
        context = deepcopy(CHANNEL_CONTEXTS[channel])
    else:
        prefix = channel.split("-")[0] if channel else ""
        context = deepcopy(PREFIX_CONTEXTS.get(prefix, DEFAULT_CONTEXT))

    context["channel"] = channel
    if spacecraft:
        context["spacecraft"] = spacecraft
    return context


def get_protocol_snippet(channel_or_prefix: str) -> str:
    """Return the protocol text for an exact channel, prefix, or safe default."""
    if channel_or_prefix in CHANNEL_CONTEXTS:
        return CHANNEL_CONTEXTS[channel_or_prefix]["protocol"]
    return PREFIX_CONTEXTS.get(channel_or_prefix, DEFAULT_CONTEXT)["protocol"]


# ===========================================================================
# CONTRACT B - Agent (Person C) -> Display (Person D)
# Emitted once per detection event. Includes reasoning, context, crew routing,
# and the deterministic policy verdict.
# ===========================================================================

EXAMPLE_GEMMA_DECISION = {
    "event_id": "evt_S1_5305",      # MUST match the detection event it responds to
    # --- from the agent/Gemma reasoning layer ---
    "severity": "medium",           # "low" | "medium" | "high"
    "confidence": 0.78,             # float 0.0 - 1.0
    "subsystem": "sensor/attitude telemetry",
    "context_summary": (
        "S-1 is treated as an attitude/state-estimation quality signal; a brief "
        "point anomaly can corrupt pointing confidence if ingested unchecked."
    ),
    "rationale": (
        "A 7.8-sigma S-1 deviation inside the known SMAP anomaly window is sharp "
        "enough to quarantine from automated trust, but the current range is not "
        "yet long enough to justify safe mode."
    ),
    "recommended_action": "isolate_channel",
    "affected_roles": [
        "flight_director",
        "adcs_officer",
        "systems_engineer",
        "science_payload_lead",
        "ground_comms",
    ],
    "crew_instructions": [
        {
            "recipient": "adcs_officer",
            "priority": "medium",
            "instruction": (
                "Cross-check S-1 against S-2, A-prefix, and D-prefix channels; "
                "remove S-1 from automated estimator trust if isolation executes."
            ),
        },
        {
            "recipient": "science_payload_lead",
            "priority": "medium",
            "instruction": (
                "Mark products generated during the flagged S-1 interval for "
                "pointing-quality review."
            ),
        },
    ],
    "next_checks": [
        "Compare S-1 against S-2 and attitude-related A/D channels.",
        "Inspect command history for events near timestep 5305.",
        "Escalate if the flagged range grows beyond 50 timesteps.",
    ],
    # --- from the deterministic policy engine (NOT the LLM) ---
    "policy_decision": "AUTONOMOUS_ACT",
    "command": "isolate_channel",   # action actually taken, or None if held/queued
    "policy_reason": "action 'isolate_channel' pre-cleared and confidence 0.78 >= 0.60",
}

GEMMA_DECISION_KEYS = set(EXAMPLE_GEMMA_DECISION.keys())
CREW_INSTRUCTION_KEYS = {"recipient", "priority", "instruction"}


def validate_crew_instruction(d: dict) -> dict:
    """Validate one targeted crew instruction item."""
    missing = CREW_INSTRUCTION_KEYS - d.keys()
    extra = d.keys() - CREW_INSTRUCTION_KEYS
    assert not missing, f"[Contract B] crew instruction MISSING keys: {missing}"
    assert not extra, f"[Contract B] crew instruction has UNEXPECTED keys: {extra}"

    assert d["recipient"] in CREW_ROLES, (
        f"[Contract B] instruction recipient must be one of {CREW_ROLES}, "
        f"got {d['recipient']!r}"
    )
    assert d["priority"] in PRIORITIES, (
        f"[Contract B] instruction priority must be one of {PRIORITIES}, "
        f"got {d['priority']!r}"
    )
    assert isinstance(d["instruction"], str) and d["instruction"].strip(), (
        "[Contract B] instruction must be a non-empty str"
    )
    return d


def make_crew_instruction(recipient: str, priority: str, instruction: str) -> dict:
    """Helper for building instruction dicts without typo-prone keys."""
    return validate_crew_instruction(
        {
            "recipient": recipient,
            "priority": priority,
            "instruction": instruction,
        }
    )


def validate_gemma_decision(d: dict) -> dict:
    """Person C calls this right before emitting a decision to Person D."""
    missing = GEMMA_DECISION_KEYS - d.keys()
    extra = d.keys() - GEMMA_DECISION_KEYS
    assert not missing, f"[Contract B] gemma decision MISSING keys: {missing}"
    assert not extra, f"[Contract B] gemma decision has UNEXPECTED keys: {extra}"

    assert isinstance(d["event_id"], str), "[Contract B] event_id must be str"
    assert d["severity"] in SEVERITIES, (
        f"[Contract B] severity must be one of {SEVERITIES}, got {d['severity']!r}"
    )

    conf = d["confidence"]
    assert isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0, (
        f"[Contract B] confidence must be a float in [0,1], got {conf!r}"
    )

    assert isinstance(d["subsystem"], str) and d["subsystem"].strip(), (
        "[Contract B] subsystem must be a non-empty str"
    )
    assert isinstance(d["context_summary"], str) and d["context_summary"].strip(), (
        "[Contract B] context_summary must be a non-empty str"
    )
    assert isinstance(d["rationale"], str) and d["rationale"].strip(), (
        "[Contract B] rationale must be a non-empty str"
    )
    assert d["recommended_action"] in ACTIONS, (
        f"[Contract B] recommended_action must be one of {ACTIONS}, "
        f"got {d['recommended_action']!r}"
    )

    roles = d["affected_roles"]
    assert isinstance(roles, list) and roles, (
        "[Contract B] affected_roles must be a non-empty list"
    )
    assert all(role in CREW_ROLES for role in roles), (
        f"[Contract B] affected_roles must only contain {CREW_ROLES}, got {roles!r}"
    )

    instructions = d["crew_instructions"]
    assert isinstance(instructions, list) and instructions, (
        "[Contract B] crew_instructions must be a non-empty list"
    )
    for item in instructions:
        validate_crew_instruction(item)
        assert item["recipient"] in roles, (
            "[Contract B] each crew instruction recipient must also appear in "
            f"affected_roles, got {item['recipient']!r}"
        )

    next_checks = d["next_checks"]
    assert isinstance(next_checks, list) and next_checks, (
        "[Contract B] next_checks must be a non-empty list"
    )
    assert all(isinstance(check, str) and check.strip() for check in next_checks), (
        "[Contract B] next_checks entries must be non-empty strings"
    )

    assert d["policy_decision"] in POLICY_DECISIONS, (
        f"[Contract B] policy_decision must be one of {POLICY_DECISIONS}, "
        f"got {d['policy_decision']!r}"
    )
    assert d["command"] is None or d["command"] in ACTIONS, (
        f"[Contract B] command must be None or one of {ACTIONS}, got {d['command']!r}"
    )
    assert isinstance(d["policy_reason"], str) and d["policy_reason"].strip(), (
        "[Contract B] policy_reason must be a non-empty str"
    )
    return d


def apply_policy(
    gemma_output: dict,
    event_id: str,
    *,
    subsystem: str = "unknown telemetry",
    context_summary: str = DEFAULT_CONTEXT["context_summary"],
    affected_roles: Optional[list[str]] = None,
    crew_instructions: Optional[list[dict]] = None,
    next_checks: Optional[list[str]] = None,
) -> dict:
    """Apply bounded-autonomy policy to the agent recommendation.

    This never trusts the reasoning layer blindly: an action only executes
    autonomously if it is pre-cleared AND confidence clears the gate.
    """
    action = gemma_output["recommended_action"]
    conf = float(gemma_output["confidence"])

    if conf < CONFIDENCE_GATE:
        decision, command = "HOLD_LOW_CONFIDENCE", None
        reason = f"confidence {conf:.2f} < {CONFIDENCE_GATE:.2f} gate; holding for review"
    elif action in AUTONOMOUS_OK:
        decision, command = "AUTONOMOUS_ACT", action
        reason = f"action '{action}' pre-cleared and confidence {conf:.2f} >= {CONFIDENCE_GATE:.2f}"
    else:
        decision, command = "QUEUE_FOR_GROUND", None
        reason = f"action '{action}' requires ground approval; queued for next contact window"

    roles = affected_roles or ["flight_director", "systems_engineer", "ground_comms"]
    instructions = crew_instructions or [
        make_crew_instruction(
            "flight_director",
            "medium",
            "Review the anomaly packet and assign subsystem ownership.",
        )
    ]
    checks = next_checks or ["Confirm whether the anomaly is transient or sustained."]

    decision_dict = {
        "event_id": event_id,
        "severity": gemma_output["severity"],
        "confidence": conf,
        "subsystem": subsystem,
        "context_summary": context_summary,
        "rationale": gemma_output["rationale"],
        "recommended_action": action,
        "affected_roles": roles,
        "crew_instructions": instructions,
        "next_checks": checks,
        "policy_decision": decision,
        "command": command,
        "policy_reason": reason,
    }
    return validate_gemma_decision(decision_dict)


# ===========================================================================
# Self-test - run `python contracts.py` to confirm examples and helpers.
# ===========================================================================

if __name__ == "__main__":
    validate_detection_event(EXAMPLE_DETECTION_EVENT)
    validate_gemma_decision(EXAMPLE_GEMMA_DECISION)

    ev = make_detection_event("S-1", "SMAP", 5305, [5300, 5741], 7.8, "zscore")
    assert ev["event_id"] == "evt_S1_5305"
    assert ev["channel_prefix"] == "S"

    context = get_channel_context("S-1", "SMAP")
    assert context["subsystem"] == "sensor/attitude telemetry"
    assert context["known_anomaly_ranges"] == [[5300, 5747]]

    fake_llm = {
        "severity": "medium",
        "confidence": 0.74,
        "rationale": "transient S-1 glitch",
        "recommended_action": "isolate_channel",
    }
    dec = apply_policy(
        fake_llm,
        ev["event_id"],
        subsystem=context["subsystem"],
        context_summary=context["context_summary"],
        affected_roles=context["affected_roles"],
        crew_instructions=[
            make_crew_instruction(
                "adcs_officer",
                "medium",
                "Cross-check redundant attitude sensors before restoring S-1 trust.",
            )
        ],
        next_checks=["Compare S-1 with S-2."],
    )
    assert dec["policy_decision"] == "AUTONOMOUS_ACT"
    assert dec["command"] == "isolate_channel"

    fake_llm_low = dict(fake_llm, confidence=0.4)
    assert apply_policy(fake_llm_low, ev["event_id"])["policy_decision"] == "HOLD_LOW_CONFIDENCE"

    fake_llm_esc = dict(fake_llm, recommended_action="escalate", confidence=0.9)
    queued = apply_policy(fake_llm_esc, ev["event_id"])
    assert queued["policy_decision"] == "QUEUE_FOR_GROUND"
    assert queued["command"] is None

    print("contracts.py self-test passed OK")
    print("  Contract A example:", EXAMPLE_DETECTION_EVENT["event_id"])
    print("  Contract B example:", EXAMPLE_GEMMA_DECISION["policy_decision"])
