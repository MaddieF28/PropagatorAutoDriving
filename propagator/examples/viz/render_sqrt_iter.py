#!/usr/bin/env python3
"""Render image files for the sqrt-iter propagator circuit.

Usage:
    python -m propagator.examples.viz.render_sqrt_iter
    python -m propagator.examples.viz.render_sqrt_iter --format svg --mode hierarchical
    python -m propagator.examples.viz.render_sqrt_iter --output-dir ./artifacts
    python -m propagator.examples.viz.render_sqrt_iter --trigger 3.0
"""

from __future__ import annotations

from propagator.circuit_renderer import CircuitRenderer
import propagator.examples.tutorial.sqrt_iter as sqrt_iter_module


if __name__ == "__main__":
    renderer = CircuitRenderer(
        module=sqrt_iter_module,
        grouped_functions=["sqrt_network", "sqrt_iter", "good_enough", "heron_step"],
        description="Render sqrt-iter circuit images (PNG/SVG)",
        default_basename="sqrt_iter",
    )

    # Build the network and capture the topology.
    trace = renderer.build_trace(
        builder_func=sqrt_iter_module.testing_sqrt_iter,
        pin_cells=[],
    )

    # Annotate cells with their runtime TMS content and premise colours.
    # This must be called after build_trace (network has already been run).
    renderer.annotate_runtime(trace)

    rendered = renderer.render(
        trace,
        output_dir=".",
        compact=True,
        basename="sqrt_iter",
        mode="hierarchical",
    )

    
    raise SystemExit(0 if rendered else 1)

