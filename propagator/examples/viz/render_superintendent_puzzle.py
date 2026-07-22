#!/usr/bin/env python3
"""Render image files for the superintendent puzzle propagator circuit.

Usage:
    python -m propagator.examples.viz.render_superintendent_puzzle
    python -m propagator.examples.viz.render_superintendent_puzzle --format svg
    python -m propagator.examples.viz.render_superintendent_puzzle --output-dir ./artifacts
"""

from __future__ import annotations

from propagator.circuit_renderer import CircuitRenderer
import propagator.examples.puzzles.superintendent_puzzle as puzzle_module


if __name__ == "__main__":
    renderer = CircuitRenderer(
        module=puzzle_module,
        grouped_functions=["multiple_dwelling"],
        description="Render superintendent puzzle circuit images (PNG/SVG)",
        default_basename="superintendent_puzzle",
        # No trigger needed: multiple_dwelling() wires everything eagerly.
    )

    # Build the network and capture the topology.
    trace = renderer.build_trace(
        builder_func=puzzle_module.multiple_dwelling,
        pin_cells=[],
    )

    # Annotate cells with their runtime TMS content and premise colours.
    # This must be called after build_trace (network has already been run).
    renderer.annotate_runtime(trace)

    # Render to image.  Annotations are picked up automatically.
    rendered = renderer.render(
        trace,
        output_dir=".",
        basename="superintendent_puzzle",
        mode="structural", #'behavioral', 'structural', 'hierarchical'
    )
    raise SystemExit(0 if rendered else 1)
