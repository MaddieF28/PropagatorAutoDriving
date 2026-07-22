"""Tests for the structured burst payload tutorial example."""

from propagator import Cell, make_log, provenance_aware_function_to_propagator_constructor, supported
from propagator.examples.tutorial.log_stream_ingestion import (
    BurstDetection,
    RecursiveBurstDetector,
    make_burst_detection,
    build_sample_stream,
    ingest,
)
from propagator.supported_values import get_support_premises, supported_p


def test_provenance_aware_constructor_preserves_supported_inputs():
    kind = Cell(name="kind")
    count = Cell(name="count")
    min_events = Cell(name="min_events")
    window_seconds = Cell(name="window_seconds")
    event_timestamp = Cell(name="event_timestamp")
    output = Cell(name="output")

    constructor = provenance_aware_function_to_propagator_constructor(make_burst_detection)
    constructor(kind, count, min_events, window_seconds, event_timestamp, output)

    premise = "test-premise"
    kind.add_content("burst")
    count.add_content(3)
    min_events.add_content(supported(3, [premise]))
    window_seconds.add_content(5.0)
    event_timestamp.add_content(4.0)

    assert supported_p(output.content)
    assert output.content.value == BurstDetection(
        kind="burst",
        count=3,
        min_events=3,
        window_seconds=5.0,
        event_timestamp=4.0,
    )
    assert get_support_premises(output.content) == [premise]


def test_burst_example_uses_structured_detection_payloads():
    _, burst_log, _ = build_sample_stream()

    assert len(burst_log.content.values) == 2

    first = burst_log.content.values[0]
    second = burst_log.content.values[1]

    assert isinstance(first.value, BurstDetection)
    assert isinstance(second.value, BurstDetection)

    assert first.value == BurstDetection(
        kind="burst",
        count=3,
        min_events=3,
        window_seconds=5.0,
        event_timestamp=4.0,
    )
    assert second.value == BurstDetection(
        kind="burst",
        count=3,
        min_events=3,
        window_seconds=5.0,
        event_timestamp=11.0,
    )


def test_lift_burst_detection_preserves_supported_provenance():
    premise = "configured-threshold"

    event_log = Cell(name="event_log")
    burst_log = Cell(name="burst_log")
    window_seconds = Cell(name="window_seconds")
    min_events = Cell(name="min_events")

    detector = RecursiveBurstDetector(
        burst_log=burst_log,
        window_seconds=window_seconds,
        min_events=min_events,
    )

    event_log.add_content(make_log())
    burst_log.add_content(make_log())
    window_seconds.add_content(5.0)
    min_events.add_content(supported(3, [premise]))

    ingest(event_log, detector, value="sensor-A", timestamp=1.0)
    ingest(event_log, detector, value="sensor-B", timestamp=2.0)
    ingest(event_log, detector, value="sensor-C", timestamp=4.0)

    detection_payload = burst_log.content.values[0].value
    assert supported_p(detection_payload)
    assert detection_payload.value == BurstDetection(
        kind="burst",
        count=3,
        min_events=3,
        window_seconds=5.0,
        event_timestamp=4.0,
    )
    assert get_support_premises(detection_payload) == [premise]