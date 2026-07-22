"""Streaming log ingestion with a single growing cell.

This example shows the intended pattern for timestamped ingestion:
- Keep one input cell containing a monotonic Log value.
- Ingest each new event as a LogEntry into that same cell.
- Derive additional monotonic facts (here: detected burst windows) in
    downstream cells using propagators.

This module includes two detector styles:
1. A direct Python detector (`detect_bursts`) for readability.
2. A recursive detector network (`RecursiveBurstDetector`) where all arithmetic
     and comparisons are performed by propagator primitives.

Run:
    python -m propagator.examples.tutorial.log_stream_ingestion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from traitlets import Callable

from propagator import (
    adder,
    Cell,
    conditional,
    conjoiner,
    constant,
    detect_bursts_in_log,
    function_to_propagator_constructor,
    gte,
    initialize_scheduler,
    lte,
    entry_to_log,
    make_log,
    make_log_entry,
    provenance_aware_function_to_propagator_constructor,
    subtractor,
    switch,
)
from propagator.log import Log, LogEntry, count_window
from propagator.supported_values import supported


DEFAULT_STREAM = [
    ("sensor-A", 1.0),
    ("sensor-B", 2.0),
    ("sensor-C", 4.0),
    ("sensor-D", 9.0),
    ("sensor-E", 10.0),
    ("sensor-F", 11.0),
]


@dataclass(frozen=True)
class BurstDetection:
    """Structured payload for a detected burst event."""

    kind: str
    count: int
    min_events: int
    window_seconds: float
    event_timestamp: float


def make_burst_detection(
    kind: str,
    count: int,
    min_events: int,
    window_seconds: float,
    event_timestamp: float,
) -> BurstDetection:
    """Create a structured burst-detection payload."""
    return BurstDetection(
        kind=kind,
        count=count,
        min_events=min_events,
        window_seconds=window_seconds,
        event_timestamp=event_timestamp,
    )

def detect_bursts_in_log(
    log: Log,
    window_seconds: float,
    min_events: int,
    payload_builder: Callable[[int, int, float, float], Any],
) -> Log:
    """Return a burst-detection log using named log-domain operators.

    payload_builder receives: (count, min_events, window_seconds, event_timestamp).
    """
    detections: list[LogEntry] = []
    for entry in log.values:
        event_timestamp = entry.timestamp
        count = count_window(log, event_timestamp, window_seconds)
        if count >= min_events:
            detections.append(
                make_log_entry(
                    payload_builder(count, min_events, window_seconds, event_timestamp),
                    event_timestamp,
                )
            )
    return make_log(detections)


def detect_bursts(events: Log, window_seconds: float, min_events: int) -> Log:
    """Return a log of burst detections discovered from an event log.

    A burst is recorded at timestamp t when at least min_events occurred in
    [t - window_seconds, t]. Detections are appended as LogEntry values and are
    themselves monotonic facts.
    """
    return detect_bursts_in_log(
        events,
        window_seconds,
        min_events,
        lambda count, threshold, window, event_timestamp: make_burst_detection(
            kind="burst",
            count=count,
            min_events=threshold,
            window_seconds=window,
            event_timestamp=event_timestamp,
        ),
    )


def _sum_cells_recursive(terms: list[Cell], out: Cell, index: int = 0) -> None:
    """Recursively build out = sum(terms[index:]) using propagator adders."""
    if index >= len(terms):
        constant(0, out)
        return

    if index == len(terms) - 1:
        zero = Cell(name=f"sum_zero_{index}")
        constant(0, zero)
        adder(terms[index], zero, out)
        return

    tail_sum = Cell(name=f"tail_sum_{index}")
    _sum_cells_recursive(terms, tail_sum, index + 1)
    adder(terms[index], tail_sum, out)


def _window_membership_indicator(
    event_ts: Cell,
    target_ts: Cell,
    window_seconds: Cell,
    out: Cell,
    suffix: str,
) -> None:
    """Build an indicator cell out in {0, 1} for event_ts in [target-window, target]."""
    diff = Cell(name=f"diff_{suffix}")
    non_negative = Cell(name=f"non_negative_{suffix}")
    within_window = Cell(name=f"within_window_{suffix}")
    in_window = Cell(name=f"in_window_{suffix}")
    zero_float = Cell(name=f"zero_float_{suffix}")
    one = Cell(name=f"one_{suffix}")
    zero_int = Cell(name=f"zero_int_{suffix}")

    subtractor(target_ts, event_ts, diff)
    constant(0.0, zero_float)
    gte(diff, zero_float, non_negative)
    lte(diff, window_seconds, within_window)
    conjoiner(non_negative, within_window, in_window)

    constant(1, one)
    constant(0, zero_int)
    conditional(in_window, one, zero_int, out)


class RecursiveBurstDetector:
    """Incrementally extends a recursive propagator network for burst detection.

    The counting math for each timestamp is implemented with recursive propagator
    composition (_sum_cells_recursive + comparison primitives), not Python sums.
    """

    def __init__(self, burst_log: Cell, window_seconds: Cell, min_events: Cell):
        self.burst_log = burst_log
        self.window_seconds = window_seconds
        self.min_events = min_events
        self.timestamp_cells: list[Cell] = []
        self._burst_increment_constructor = function_to_propagator_constructor(
            entry_to_log
        )
        self._payload_constructor = provenance_aware_function_to_propagator_constructor(
            make_burst_detection
        )
        self._log_entry_constructor = provenance_aware_function_to_propagator_constructor(
            make_log_entry
        )

    def add_event_timestamp(self, timestamp: float) -> None:
        """Add one event timestamp and recursively build burst logic for it."""
        ts_cell = Cell(name=f"event_ts_{len(self.timestamp_cells)}")
        constant(float(timestamp), ts_cell)
        self.timestamp_cells.append(ts_cell)

        indicators: list[Cell] = []
        for i, event_ts in enumerate(self.timestamp_cells):
            indicator = Cell(name=f"indicator_{len(self.timestamp_cells) - 1}_{i}")
            _window_membership_indicator(
                event_ts,
                ts_cell,
                self.window_seconds,
                indicator,
                suffix=f"{len(self.timestamp_cells) - 1}_{i}",
            )
            indicators.append(indicator)

        count = Cell(name=f"count_{len(self.timestamp_cells) - 1}")
        _sum_cells_recursive(indicators, count)

        is_burst = Cell(name=f"is_burst_{len(self.timestamp_cells) - 1}")
        gte(count, self.min_events, is_burst)

        kind = Cell(name=f"detection_kind_{len(self.timestamp_cells) - 1}")
        payload = Cell(name=f"detection_payload_{len(self.timestamp_cells) - 1}")
        detection = Cell(name=f"detection_entry_{len(self.timestamp_cells) - 1}")

        constant("burst", kind)
        self._payload_constructor(
            kind,
            count,
            self.min_events,
            self.window_seconds,
            ts_cell,
            payload,
        )
        self._log_entry_constructor(payload, ts_cell, detection)

        gated_detection = Cell(name=f"gated_detection_{len(self.timestamp_cells) - 1}")
        switch(is_burst, detection, gated_detection)
        self._burst_increment_constructor(gated_detection, self.burst_log)


def build_network() -> tuple[Cell, Cell, RecursiveBurstDetector]:
    """Create cells and wiring for event ingestion + burst detection."""
    event_log = Cell(name="event_log")
    window_seconds = Cell(name="window_seconds")
    min_events = Cell(name="min_events")

    burst_log = Cell(name="burst_log")

    detector = RecursiveBurstDetector(
        burst_log=burst_log,
        window_seconds=window_seconds,
        min_events=min_events,
    )

    window_seconds.add_content(5.0)
    min_events.add_content(3)
    event_log.add_content(make_log())
    burst_log.add_content(make_log())

    return event_log, burst_log, detector


def ingest(
    event_log: Cell,
    detector: RecursiveBurstDetector,
    *,
    value: str,
    timestamp: float,
) -> None:
    """Ingest one new event as incremental content into the same input cell."""
    event_log.add_content(make_log_entry(value=value, timestamp=timestamp))
    detector.add_event_timestamp(timestamp)


def build_sample_stream(
    stream: list[tuple[str, float]] | None = None,
) -> tuple[Cell, Cell, RecursiveBurstDetector]:
    """Build the tutorial network and ingest a representative stream.

    This helper exists for visualization and tests: it expands the recursive
    detector network without printing intermediate output.
    """
    event_log, burst_log, detector = build_network()
    for value, timestamp in stream or DEFAULT_STREAM:
        ingest(event_log, detector, value=value, timestamp=timestamp)
    return event_log, burst_log, detector


def demo() -> None:
    """Demonstrate active ingestion over time."""
    event_log, burst_log, detector = build_network()

    for value, timestamp in DEFAULT_STREAM:
        ingest(event_log, detector, value=value, timestamp=timestamp)
        print(f"ingested: {value}@{timestamp}")
        print(f"  events: {event_log.content}")
        print(f"  bursts: {burst_log.content}")


if __name__ == "__main__":
    initialize_scheduler()
    demo()
