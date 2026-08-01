from new_scripts.agent import MissionAgent
from new_scripts.contracts import get_channel_context, make_detection_event, validate_gemma_decision


def test_s1_context_routes_medium_anomaly_to_attitude_and_science_roles():
    event = make_detection_event("S-1", "SMAP", 5305, [5300, 5305], 7.8, "zscore")

    decision = MissionAgent().decide(event)

    validate_gemma_decision(decision)
    assert decision["subsystem"] == "sensor/attitude telemetry"
    assert decision["recommended_action"] == "isolate_channel"
    assert decision["policy_decision"] == "AUTONOMOUS_ACT"
    assert decision["command"] == "isolate_channel"
    assert "adcs_officer" in decision["affected_roles"]
    assert "science_payload_lead" in decision["affected_roles"]
    assert any(item["recipient"] == "science_payload_lead" for item in decision["crew_instructions"])


def test_s1_critical_sustained_anomaly_requires_ground_approval():
    event = make_detection_event("S-1", "SMAP", 5550, [5300, 5550], 9.5, "zscore")

    decision = MissionAgent().decide(event)

    validate_gemma_decision(decision)
    assert decision["severity"] == "high"
    assert decision["recommended_action"] == "safe_mode"
    assert decision["policy_decision"] == "QUEUE_FOR_GROUND"
    assert decision["command"] is None
    assert any("ground" in item["instruction"].lower() for item in decision["crew_instructions"])


def test_s1_context_includes_known_label_window():
    context = get_channel_context("S-1", "SMAP")

    assert context["known_anomaly_ranges"] == [[5300, 5747]]
    assert context["known_anomaly_class"] == "point"
