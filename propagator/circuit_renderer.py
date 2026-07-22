"""Generic propagator circuit renderer utility.

This module provides reusable utilities for rendering propagator networks
from any example program. It handles DOT generation, image rendering, and
command-line option parsing.

Example usage in your own program:

    from propagator.circuit_renderer import CircuitRenderer
    import my_example_module

    renderer = CircuitRenderer(
        module=my_example_module,
        grouped_functions=["my_network", "my_helper", "nested_function"]
    )
    
    # Render from CLI
    renderer.run_cli()
    
    # Or render programmatically
    trace = renderer.build_trace(
        builder_func=my_example_module.my_network,
        pin_cells=[x, y],
        trigger_value=1.5,
    )
    renderer.render(trace, output_dir=".", basename="my_network", mode="hierarchical")
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from propagator import Cell, capture_circuit
from propagator.circuit_viz import CircuitTrace


class CircuitRenderer:
    """Generic renderer for propagator circuit visualizations."""

    def __init__(
        self,
        module: types.ModuleType,
        grouped_functions: List[str],
        description: str = "Render propagator circuit images",
        default_basename: str = "circuit",
        default_trigger: Optional[float] = None,
    ):
        """Initialize the renderer.

        Args:
            module: Python module containing the network building functions.
            grouped_functions: List of function names to group hierarchically.
            description: Description for CLI help text.
            default_basename: Default output file base name (without extension).
            default_trigger: Default trigger value used by ``--trigger`` on the
                CLI.  See ``build_trace`` for an explanation of what it does.
        """
        self.module = module
        self.grouped_functions = grouped_functions
        self.description = description
        self.default_basename = default_basename
        self.default_trigger = default_trigger

    def build_trace(
        self,
        builder_func: Callable,
        pin_cells: Optional[list] = None,
        trigger_value: Optional[float] = None,
        builder_kwargs: Optional[dict] = None,
    ) -> CircuitTrace:
        """Build and capture a circuit trace.

        Args:
            builder_func: Function that builds the network.  Called as
                ``builder_func(*pin_cells, **builder_kwargs)`` when *pin_cells*
                is non-empty, or as ``builder_func(**builder_kwargs)`` when
                *pin_cells* is ``None`` or empty (useful for builders that
                create their own cells internally).
            pin_cells: Cell objects to pin so they always appear as labelled
                nodes in the diagram.  Pass ``None`` or ``[]`` for builders
                that create their own cells.
            trigger_value: A value added to the first pin cell to make lazy
                ``compound_propagator`` networks fire during tracing.  Most
                physics / iterative examples (``fall_duration``, ``sqrt_iter``)
                build their internal wiring lazily — the inner cells and
                primitives only exist after an input fires.  Providing any
                representative value here (e.g. ``3.0`` for time in seconds)
                causes the scheduler to flush queued propagators *inside* the
                capture context so the full network topology is recorded.  The
                exact value does not affect the graph structure.  Omit (or pass
                ``None``) for networks whose full structure is wired up
                eagerly (e.g. CSP problems built with ``one_of``/``require``).
            builder_kwargs: Optional keyword arguments forwarded to
                *builder_func*.

        Returns:
            CircuitTrace object with the captured network.
        """
        import propagator.primitives as _primitives
        from propagator import initialize_scheduler, run

        if builder_kwargs is None:
            builder_kwargs = {}
        pin = list(pin_cells) if pin_cells else []

        # Start from a clean scheduler state so previous runs don't interfere.
        initialize_scheduler()

        # Patch both the example module's namespace (for names imported there)
        # AND propagator.primitives (for primitives called inside composite
        # helpers like product/quadratic whose globals live in that module).
        with capture_circuit(
            target_namespaces=[self.module.__dict__, _primitives.__dict__],
            pin_cells=pin,
            group_function_names=self.grouped_functions,
        ) as trace:
            if pin:
                builder_func(*pin, **builder_kwargs)
            else:
                builder_func(**builder_kwargs)

            if trigger_value is not None and pin:
                # Seed the first pin cell, then flush the scheduler so that
                # any compound_propagator callbacks fire — building their
                # internal networks — while the primitives are still patched.
                pin[0].add_content(trigger_value)
                try:
                    run()
                except Exception:
                    # Contradictions or other runtime errors during the trigger
                    # run are expected (e.g. TMS nogoods).  We only care about
                    # the wiring, not the computed values.
                    pass

        return trace

    def render(
        self,
        trace: CircuitTrace,
        output_dir: Union[str, Path] = ".",
        basename: str = "circuit",
        mode: str = "hierarchical",
        image_format: str = "png",
        compact: bool = True,
    ) -> Optional[Path]:
        """Render a trace to an image file.
        
        Args:
            trace: CircuitTrace object to render.
            output_dir: Directory for output files.
            basename: Base file name without extension.
            mode: Rendering mode ('behavioral', 'structural', 'hierarchical').
            image_format: Image format ('png' or 'svg').
            compact: Whether to use compact rendering.
            
        Returns:
            Path to rendered image, or None if rendering failed.
        """
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        dot_path = out_dir / f"{basename}.dot"
        image_path = out_dir / f"{basename}.{image_format}"

        trace.write_dot(dot_path, compact=compact, mode=mode)

        try:
            rendered = trace.render(
                image_path, mode=mode, image_format=image_format, compact=compact
            )
            return Path(rendered)
        except RuntimeError as exc:
            print(str(exc))
            print("Attempting Graphviz CLI fallback via 'dot'...")

            if self._render_with_dot_cli(dot_path, image_path, image_format):
                return image_path

            print("Could not render image. DOT file is available for later rendering:")
            print(dot_path)
            return None

    def _render_with_dot_cli(
        self, dot_path: Path, image_path: Path, image_format: str
    ) -> bool:
        """Render using Graphviz CLI if available."""
        if shutil.which("dot") is None:
            return False

        cmd = ["dot", f"-T{image_format}", str(dot_path), "-o", str(image_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as exc:
            print("dot command failed:")
            print(exc.stderr.strip() or exc.stdout.strip() or str(exc))
            return False

    def run_cli(self, builder_func: Callable, pin_cell_names: List[str]) -> int:
        """Run a command-line interface for rendering.

        Args:
            builder_func: Function that builds the network.
            pin_cell_names: Names of cells to create and pin (e.g. ``['x', 'answer']``).
                Pass an empty list for builders that create their own cells internally.

        Returns:
            Exit code (0 for success, 1 for failure).
        """
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument(
            "--mode",
            choices=["behavioral", "structural", "hierarchical"],
            default="hierarchical",
            help="Visualization mode",
        )
        parser.add_argument(
            "--format", choices=["png", "svg"], default="png", help="Output image format"
        )
        parser.add_argument(
            "--output-dir", default=".", help="Directory for output files"
        )
        parser.add_argument(
            "--basename",
            default=self.default_basename,
            help="Base file name without extension",
        )
        parser.add_argument(
            "--trigger", type=float, default=self.default_trigger,
            metavar="VALUE",
            help="Value added to the first pin cell to trigger lazy compound "
                 "propagators.  Required for networks built with "
                 "compound_propagator (e.g. fall_duration, sqrt_iter).",
        )
        parser.add_argument(
            "--annotate", action="store_true", default=False,
            help="Annotate cells with their runtime content and premise colours "
                 "before rendering.  Adds TMS values and colour legend to the diagram.",
        )
        args = parser.parse_args()

        pin_cells = [Cell(name=name) for name in pin_cell_names]

        trace = self.build_trace(
            builder_func=builder_func,
            pin_cells=pin_cells or None,
            trigger_value=args.trigger,
        )

        if args.annotate:
            self.annotate_runtime(trace)

        rendered = self.render(
            trace,
            output_dir=args.output_dir,
            basename=args.basename,
            mode=args.mode,
            image_format=args.format,
        )

        if rendered:
            out_dir = Path(args.output_dir)
            dot_path = out_dir / f"{args.basename}.dot"
            print(f"wrote image: {rendered}")
            print(f"wrote dot:   {dot_path}")
            return 0
        else:
            return 1

    def annotate_runtime(self, trace: CircuitTrace) -> None:
        """Attach runtime content annotations to a trace built with :meth:`build_trace`.

        ``build_trace`` already runs the network internally; call this method
        immediately afterwards to enrich the diagram with cell values and premise
        colours.  The annotations are stored on each :class:`CellNode` and are
        picked up automatically by the DOT renderers.

        After this call, use ``trace.premise_colors`` to access the
        ``PremiseColorMap`` (``id(premise) → colour``).

        Example::

            trace = renderer.build_trace(my_fn, pin_cells=[x, y], trigger_value=1.0)
            renderer.annotate_runtime(trace)
            renderer.render(trace, ...)
        """
        return trace.annotate_runtime()
