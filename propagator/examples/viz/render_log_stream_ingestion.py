#!/usr/bin/env python3
"""Render image files for the log-stream burst detector circuit.

Usage:
    python -m propagator.examples.viz.render_log_stream_ingestion
    python -m propagator.examples.viz.render_log_stream_ingestion --format svg
    python -m propagator.examples.viz.render_log_stream_ingestion --output-dir ./artifacts
"""

from __future__ import annotations

from propagator.circuit_renderer import CircuitRenderer
import propagator.examples.tutorial.log_stream_ingestion as log_stream_module


if __name__ == "__main__":
    renderer = CircuitRenderer(
        module=log_stream_module,
        grouped_functions=[
            "build_network",
            "build_sample_stream",
            "ingest",
            "_sum_cells_recursive",
            "_window_membership_indicator",
        ],
        description="Render log-stream burst detector circuit images (PNG/SVG)",
        default_basename="log_stream_ingestion",
    )

    trace = renderer.build_trace(
        builder_func=log_stream_module.build_sample_stream,
        pin_cells=[],
    )

    renderer.annotate_runtime(trace)
    rendered = renderer.render(
        trace,
        output_dir=".",
        basename="log_stream_ingestion",
        mode="hierarchical",
    )

    raise SystemExit(0 if rendered else 1)