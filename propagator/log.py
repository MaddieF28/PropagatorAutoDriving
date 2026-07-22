
"""Monotonic log values for propagator cells.

Logs are treated as a join-semilattice:
- Information only grows (new entries can be added, existing entries are stable).
- Merge is commutative and associative.
- Duplicate facts collapse by entry key.

This lets a single cell ingest timestamped events over time without creating a
new cell per event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .nothing import nothing
from .merge import assign_equivalent_operation, assign_merge_operation


@dataclass(frozen=True)
class LogEntry:
    """A single timestamped fact in a log."""
    value: Any
    timestamp: float

    @property
    def key(self) -> tuple[float, Any]:
        """Identity key used for idempotent set semantics."""
        return (self.timestamp, self.value)


@dataclass(frozen=True)
class Log:
    """Immutable, deduplicated, timestamp-ordered log."""
    values: tuple[LogEntry, ...]

    def __post_init__(self) -> None:
        # Canonicalize representation so equality and merge are deterministic.
        unique: dict[tuple[float, Any], LogEntry] = {}
        for entry in self.values:
            unique[entry.key] = entry
        ordered = tuple(sorted(unique.values(), key=lambda e: (e.timestamp, repr(e.value))))
        object.__setattr__(self, "values", ordered)

    def __repr__(self):
        return f"Log(values={self.values})"

    def contains(self, entry: LogEntry) -> bool:
        return entry in self.values

    def count(self) -> int:
        return len(self.values)

    def timestamps(self) -> tuple[float, ...]:
        return tuple(entry.timestamp for entry in self.values)

    def entries_between(self, start: float, end: float) -> tuple[LogEntry, ...]:
        return tuple(entry for entry in self.values if start <= entry.timestamp <= end)

    def entries_after(self, timestamp: float) -> tuple[LogEntry, ...]:
        return tuple(entry for entry in self.values if entry.timestamp > timestamp)

    def window(self, end: float, duration: float) -> "Log":
        return make_log(self.entries_between(end - duration, end))

    def latest(self) -> LogEntry | None:
        if not self.values:
            return None
        return self.values[-1]

    def latest_before(self, timestamp: float) -> LogEntry | None:
        candidates = [entry for entry in self.values if entry.timestamp <= timestamp]
        if not candidates:
            return None
        return candidates[-1]

    def filter(self, predicate: Callable[[LogEntry], bool]) -> "Log":
        return make_log(entry for entry in self.values if predicate(entry))

    def map(self, mapper: Callable[[LogEntry], LogEntry]) -> "Log":
        return make_log(mapper(entry) for entry in self.values)

    def payloads(self) -> tuple[Any, ...]:
        return tuple(entry.value for entry in self.values)


def log_p(value: Any) -> bool:
    """Predicate for merge/generic dispatch."""
    return isinstance(value, Log)


def log_entry_p(value: Any) -> bool:
    """Predicate for merge/generic dispatch."""
    return isinstance(value, LogEntry)


def _normalize_values(values: Iterable[LogEntry]) -> tuple[LogEntry, ...]:
    return tuple(values)


def merge_logs(left: Log, right: Log) -> Log:
    """Join operator for logs: set union over entries."""
    return Log(values=left.values + right.values)


def merge_log_with_entry(log: Log, entry: LogEntry) -> Log:
    """Incremental ingestion: add one entry into an existing log."""
    return Log(values=log.values + (entry,))


def log_equivalent(left: Log, right: Log) -> bool:
    """Semantic equivalence: same canonical entries."""
    return left.values == right.values


def make_log_entry(value: Any, timestamp: float) -> LogEntry:
    """Create a single timestamped entry."""
    return LogEntry(value=value, timestamp=timestamp)


def make_log(values: Iterable[LogEntry] | None = None) -> Log:
    """Create a log from zero or more entries."""
    if values is None:
        return Log(values=tuple())
    return Log(values=_normalize_values(values))


def append_log_entry(log: Log, value: Any, timestamp: float) -> Log:
    """Functional append helper for one-event-at-a-time ingestion."""
    return merge_log_with_entry(log, make_log_entry(value=value, timestamp=timestamp))


def count_log(log: Log) -> int:
    """Return the number of entries in a log."""
    return log.count()


def timestamps(log: Log) -> tuple[float, ...]:
    """Return all entry timestamps in canonical order."""
    return log.timestamps()


def entries_between(log: Log, start: float, end: float) -> Log:
    """Return the sublog whose timestamps fall in [start, end]."""
    return make_log(log.entries_between(start, end))


def window_log(log: Log, end: float, duration: float) -> Log:
    """Return the sublog covering [end - duration, end]."""
    return log.window(end, duration)


def count_window(log: Log, end: float, duration: float) -> int:
    """Count entries in the window [end - duration, end]."""
    return window_log(log, end, duration).count()


def latest(log: Log) -> LogEntry | None:
    """Return the latest entry in the log, or None if empty."""
    return log.latest()


def latest_before(log: Log, timestamp: float) -> LogEntry | None:
    """Return the latest entry whose timestamp is <= timestamp."""
    return log.latest_before(timestamp)


def latest_payload(log: Log) -> Any | None:
    """Return the payload of the latest entry, or None if empty."""
    entry = latest(log)
    return None if entry is None else entry.value


def filter_log(log: Log, predicate: Callable[[LogEntry], bool]) -> Log:
    """Return the sublog of entries matching predicate."""
    return log.filter(predicate)


def map_log(log: Log, mapper: Callable[[LogEntry], LogEntry]) -> Log:
    """Map entries to entries while preserving log canonicalization."""
    return log.map(mapper)


def filter_after_timestamp(log: Log, threshold: float) -> Log:
    """Return entries with timestamp strictly greater than threshold."""
    return make_log(entry for entry in log.values if entry.timestamp > threshold)


def filter_before_timestamp(log: Log, threshold: float) -> Log:
    """Return entries with timestamp less than or equal to threshold."""
    return make_log(entry for entry in log.values if entry.timestamp <= threshold)


def map_payload_values(log: Log, mapper: Callable[[Any], Any]) -> Log:
    """Map payload values while preserving entry timestamps."""
    return make_log(
        make_log_entry(mapper(entry.value), entry.timestamp)
        for entry in log.values
    )




def singleton_log(entry: LogEntry) -> Log:
    """Create a one-entry log increment."""
    return make_log([entry])


def entry_to_log(entry_like: Any) -> Log:
    """Convert an entry-shaped value into a log increment.

    Accepts plain LogEntry values as well as Supported/TMS-wrapped LogEntry
    values and returns a Log that can be merged directly into a log cell.
    """
    if entry_like is nothing:
        return make_log()

    if isinstance(entry_like, Log):
        return entry_like

    if isinstance(entry_like, LogEntry):
        return singleton_log(entry_like)

    # Local imports avoid creating new module cycles at import time.
    from .supported_values import merge_supports, supported, supported_p
    from .tms import make_tms, tms_p

    def merge_support_into_payload(payload: Any, support_source: Any) -> Any:
        if supported_p(payload):
            return supported(payload.value, merge_supports(support_source, payload))
        if tms_p(payload):
            merged_values = []
            for branch in payload.values:
                if supported_p(branch):
                    merged_values.append(supported(branch.value, merge_supports(support_source, branch)))
                else:
                    merged_values.append(supported(branch, support_source.support))
            return make_tms(merged_values)
        return supported(payload, support_source.support)

    if supported_p(entry_like) and isinstance(entry_like.value, LogEntry):
        entry = entry_like.value
        normalized_payload = merge_support_into_payload(entry.value, entry_like)
        return singleton_log(make_log_entry(normalized_payload, entry.timestamp))

    if tms_p(entry_like):
        normalized_entries: list[LogEntry] = []
        for branch in entry_like.values:
            branch_log = entry_to_log(branch)
            normalized_entries.extend(branch_log.values)
        return make_log(normalized_entries)

    raise TypeError(f"Unsupported log increment type: {type(entry_like).__name__}")


def normalize_log_increment(entry_like: Any) -> Log:
    """Backward-compatible alias for entry_to_log."""
    return entry_to_log(entry_like)


# Register merge/equivalent handlers for log values.
assign_merge_operation(merge_logs, log_p, log_p)
assign_merge_operation(merge_log_with_entry, log_p, log_entry_p)
assign_merge_operation(lambda entry, log: merge_log_with_entry(log, entry), log_entry_p, log_p)
assign_equivalent_operation(log_equivalent, log_p, log_p)

