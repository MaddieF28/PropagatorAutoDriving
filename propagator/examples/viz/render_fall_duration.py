#!/usr/bin/env python3
"""Render image files for the fall-duration propagator circuit.

Usage:
    python -m propagator.examples.viz.render_fall_duration
    python -m propagator.examples.viz.render_fall_duration --format svg --mode hierarchical
    python -m propagator.examples.viz.render_fall_duration --output-dir ./artifacts
    python -m propagator.examples.viz.render_fall_duration --trigger 3.0
"""

from __future__ import annotations

from propagator.circuit_renderer import CircuitRenderer
import propagator.examples.tutorial.fall_duration_basic_dependencies as fall_duration_module


if __name__ == "__main__":
    renderer = CircuitRenderer(
        module=fall_duration_module,
        grouped_functions=["fall_duration", "similar_triangles_height"],
        description="Render fall-duration circuit images (PNG/SVG)",
        default_basename="fall_duration",
    )

    trace = renderer.build_trace(
        builder_func=fall_duration_module.testing_fall_duration,
        pin_cells=[],
    )

    renderer.annotate_runtime(trace)
    rendered = renderer.render(
        trace,
        output_dir=".",
        basename="fall_duration",
        mode="hierarchical",
    )

    raise SystemExit(0 if rendered else 1)
