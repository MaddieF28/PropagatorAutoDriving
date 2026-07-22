"""Circuit-style visualization for propagator networks.

This module traces propagator constructor calls and produces a compact
operation graph. The compaction step contracts anonymous intermediate cells
so long-running networks do not explode into one node per cell state.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, Union


@dataclass
class CellNode:
    key: int
    label: str
    keep: bool
    annotation: Optional["CellAnnotation"] = None


@dataclass
class OpNode:
    key: int
    name: str
    label: str
    group_path: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Edge:
    src: Tuple[str, int]
    dst: Tuple[str, int]
    label: str = ""


@dataclass
class CellAnnotation:
    """Runtime content annotation attached to a CellNode after annotate_runtime()."""
    #: One of "nothing" | "plain" | "interval" | "supported" | "tms".
    content_type: str
    #: Compact text shown beneath the cell name in the diagram.
    display_value: str
    #: Fill colour derived from premise palette, or "" for default.
    fill_color: str
    #: (value_str, is_believed) pairs — populated for TMS cells only.
    tms_branches: List[Tuple[str, bool]]
    #: Human-readable premise descriptions for tooltip / legend.
    premise_labels: List[str]


#: Maps ``id(premise)`` to an HTML colour string.  Built by annotate_runtime().
PremiseColorMap = Dict[int, str]


def _slug(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum():
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "node"


# ── Premise colour palette ────────────────────────────────────────────────────

_PREMISE_PALETTE: List[str] = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]

_PLAIN_CONTENT_FILL = "#f0f0f0"
_INTERVAL_CONTENT_FILL = "#e8f4fd"


def _build_premise_color_map(premises: List[Any]) -> PremiseColorMap:
    """Assign a stable colour from *_PREMISE_PALETTE* to each premise object."""
    return {id(p): _PREMISE_PALETTE[i % len(_PREMISE_PALETTE)] for i, p in enumerate(premises)}


def _format_float(v: Any) -> str:
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


def _classify_content(
    content: Any,
    premise_colors: PremiseColorMap,
    cell_obj: Any = None,
) -> Optional["CellAnnotation"]:
    """Inspect *content* and return a CellAnnotation, or None if content is absent."""
    # Lazy imports to avoid circular deps at module load time.
    try:
        from .nothing import nothing_p as _nothing_p
    except ImportError:
        _nothing_p = lambda x: False
    try:
        from .intervals import Interval as _Interval
    except ImportError:
        _Interval = None
    try:
        from .supported_values import Supported as _Supported, _unwrap_support as _unwrap
    except ImportError:
        _Supported = None
        _unwrap = None
    try:
        from .tms import Tms as _Tms, tms_query as _tms_query, premise_in as _premise_in
    except ImportError:
        _Tms = None
        _tms_query = None
        _premise_in = None

    if _nothing_p(content):
        return CellAnnotation("nothing", "", "", [], [])

    # ── TMS ────────────────────────────────────────────────────────────────────
    if _Tms is not None and isinstance(content, _Tms):
        believed = _tms_query(content)
        branches: List[Tuple[str, bool]] = []
        for sv in content.values:
            if _Supported is not None and isinstance(sv, _Supported):
                vs = _format_float(sv.value) if isinstance(sv.value, (int, float)) else str(sv.value)
                is_bel = all(_premise_in(p) for p in _unwrap(sv.support))
                branches.append((vs, is_bel))
        display, fill, premise_labels = "?", "", []
        if _Supported is not None and isinstance(believed, _Supported):
            display = _format_float(believed.value) if isinstance(believed.value, (int, float)) else str(believed.value)
            premises = _unwrap(believed.support)
            leaf = next(
                (p for p in premises
                 if getattr(p, "value_if_chosen", None) == believed.value
                 and getattr(p, "output_cell", None) is cell_obj),
                premises[0] if premises else None,
            )
            fill = premise_colors.get(id(leaf), "") if leaf is not None else ""
            premise_labels = [p.describe() if hasattr(p, "describe") else repr(p) for p in premises]
        return CellAnnotation("tms", display, fill, branches, premise_labels)

    # ── Supported ──────────────────────────────────────────────────────────────
    if _Supported is not None and isinstance(content, _Supported):
        premises = _unwrap(content.support)
        val = content.value
        if _Interval is not None and isinstance(val, _Interval):
            display = f"[{_format_float(val.low)}, {_format_float(val.high)}]"
        else:
            display = _format_float(val) if isinstance(val, (int, float)) else str(val)
        leaf = next(
            (p for p in premises
             if getattr(p, "value_if_chosen", None) == val
             and getattr(p, "output_cell", None) is cell_obj),
            premises[0] if premises else None,
        )
        fill = premise_colors.get(id(leaf), "") if leaf is not None else ""
        premise_labels = [p.describe() if hasattr(p, "describe") else repr(p) for p in premises]
        return CellAnnotation("supported", display, fill, [], premise_labels)

    # ── Interval ───────────────────────────────────────────────────────────────
    if _Interval is not None and isinstance(content, _Interval):
        display = f"[{_format_float(content.low)}, {_format_float(content.high)}]"
        return CellAnnotation("interval", display, _INTERVAL_CONTENT_FILL, [], [])

    # ── Plain value ────────────────────────────────────────────────────────────
    display = str(content)
    if len(display) > 24:
        display = display[:21] + "\u2026"
    return CellAnnotation("plain", display, _PLAIN_CONTENT_FILL, [], [])


def _annotation_dot_attrs(
    label: str,
    ann: Optional["CellAnnotation"],
    *,
    is_param: bool = False,
    shared: bool = False,
) -> Tuple[str, str]:
    """Return ``(label_attr, style_attr)`` DOT fragments for a cell node.

    *label_attr* is the full ``label=\u2026`` value (HTML or quoted string).
    *style_attr* is the remaining style string (fill, peripheries, penwidth).
    """
    penwidth = "1.4" if shared else "1.0"
    default_fill = "#fff6d5" if is_param else "#ffffff"

    if ann is None or ann.content_type == "nothing":
        style = f'style="filled", fillcolor="{default_fill}", penwidth={penwidth}'
        return f'"{label}"', style

    fill = ann.fill_color or default_fill
    peripheries = 2 if ann.content_type == "tms" else 1

    if ann.content_type == "tms" and ann.tms_branches:
        believed = [v for v, b in ann.tms_branches if b]
        others = [v for v, b in ann.tms_branches if not b]
        bv = f"<B>{believed[0]}</B>" if believed else "?"
        if others:
            ov = ", ".join(others[:4]) + ("\u2026" if len(others) > 4 else "")
            lbl = (f'<<B>{label}</B><BR/>'
                   f'{bv}<BR/>'
                   f'<FONT POINT-SIZE="8" COLOR="#777777">{ov}</FONT>>')
        else:
            lbl = f'<<B>{label}</B><BR/>{bv}>'
    elif ann.display_value:
        lbl = f'<{label}<BR/><FONT POINT-SIZE="9">{ann.display_value}</FONT>>'
    else:
        lbl = f'"{label}"'

    tooltip = ""
    if ann.premise_labels:
        tip = "; ".join(ann.premise_labels[:5]).replace('"', "'")
        if len(ann.premise_labels) > 5:
            tip += "\u2026"
        tooltip = f', tooltip="{tip}"'

    style = (
        f'style="filled", fillcolor="{fill}", '
        f'peripheries={peripheries}, penwidth={penwidth}{tooltip}'
    )
    return lbl, style


class CircuitTrace:
    """Collects propagator wiring and exports a compact DOT graph."""

    def __init__(self) -> None:
        self._cell_seq = 0
        self._op_seq = 0
        self._cell_ids: Dict[int, int] = {}
        self.cells: Dict[int, CellNode] = {}
        self.ops: Dict[int, OpNode] = {}
        self.edges: Set[Edge] = set()
        self._kept_cell_objs: Set[int] = set()
        self._param_cell_tags: Dict[int, Set[Tuple[str, str]]] = {}
        self._group_instances: Dict[str, _GroupInstance] = {}
        self._cell_objs: Dict[int, Any] = {}          # cell_key → live cell object
        self._premise_colors: PremiseColorMap = {}
        self._premise_color_labels: Dict[int, str] = {}

    def pin_cells(self, cells: Iterable[Any]) -> None:
        for cell in cells:
            self._kept_cell_objs.add(id(cell))

    def _is_named_cell(self, cell: Any) -> bool:
        return bool(getattr(cell, "name", None))

    def _cell_label(self, cell: Any) -> str:
        name = getattr(cell, "name", None)
        if name:
            return str(name)
        role = getattr(cell, "role", None)
        if role:
            return str(role)
        return "cell"

    def _cell_key(self, cell: Any) -> int:
        obj_id = id(cell)
        if obj_id not in self._cell_ids:
            self._cell_seq += 1
            key = self._cell_seq
            self._cell_ids[obj_id] = key
            keep = self._is_named_cell(cell) or (obj_id in self._kept_cell_objs)
            self.cells[key] = CellNode(key=key, label=self._cell_label(cell), keep=keep)
            self._cell_objs[key] = cell
        return self._cell_ids[obj_id]

    def _op_key(self, name: str, label: str) -> int:
        self._op_seq += 1
        key = self._op_seq
        self.ops[key] = OpNode(key=key, name=name, label=label)
        return key

    def add_operation(
        self,
        name: str,
        inputs: Sequence[Any],
        outputs: Sequence[Any],
        *,
        label: Optional[str] = None,
        group_path: Optional[Tuple[str, ...]] = None,
    ) -> None:
        op_key = self._op_key(name=name, label=label or name)
        if group_path:
            self.ops[op_key].group_path = group_path
        op_ref = ("op", op_key)

        for index, cell in enumerate(inputs, start=1):
            cell_key = self._cell_key(cell)
            self.edges.add(Edge(src=("cell", cell_key), dst=op_ref, label=f"in{index}"))

        for index, cell in enumerate(outputs, start=1):
            cell_key = self._cell_key(cell)
            self.edges.add(Edge(src=op_ref, dst=("cell", cell_key), label=f"out{index}"))

    def register_param_cell(self, cell: Any, group_name: str, param_name: str) -> None:
        """Record that a cell is a parent-function parameter for hierarchical views."""
        cell_key = self._cell_key(cell)
        self._param_cell_tags.setdefault(cell_key, set()).add((group_name, param_name))

    def register_group_instance(self, token: str, name: str, label: str) -> None:
        self._group_instances[token] = _GroupInstance(name=name, label=label)

    def compact(self) -> None:
        """Contract anonymous pass-through cells into direct op-to-op edges.

        Also prunes isolated op nodes (e.g. internal ``inverter`` calls inside
        ``conditional``) whose every adjacent cell is anonymous and has no
        other connections in the traced graph.

        Mutates ``self.cells``, ``self.ops``, and ``self.edges`` in-place.
        """
        changed = True
        while changed:
            changed = False
            for cell_key, cell in list(self.cells.items()):
                if cell.keep:
                    continue

                in_edges = [e for e in self.edges if e.dst == ("cell", cell_key)]
                out_edges = [e for e in self.edges if e.src == ("cell", cell_key)]

                if len(in_edges) == 1 and len(out_edges) == 1:
                    in_edge = in_edges[0]
                    out_edge = out_edges[0]

                    if in_edge.src[0] == "op" and out_edge.dst[0] == "op":
                        merged_label = ""
                        if in_edge.label or out_edge.label:
                            merged_label = f"{in_edge.label}->{out_edge.label}".strip("-")

                        self.edges.discard(in_edge)
                        self.edges.discard(out_edge)
                        self.edges.add(Edge(src=in_edge.src, dst=out_edge.dst, label=merged_label))
                        del self.cells[cell_key]
                        changed = True
                        break

        # Prune isolated op nodes: ops where every adjacent cell is anonymous
        # (keep=False) and has degree 1 (connected to this op only).  These
        # arise from internal helpers such as the inverter inside conditional().
        changed = True
        while changed:
            changed = False
            for op_key in list(self.ops):
                op_in_edges = [e for e in self.edges if e.dst == ("op", op_key)]
                op_out_edges = [e for e in self.edges if e.src == ("op", op_key)]

                adjacent_cell_keys: Set[int] = set()
                for e in op_in_edges:
                    if e.src[0] == "cell":
                        adjacent_cell_keys.add(e.src[1])
                for e in op_out_edges:
                    if e.dst[0] == "cell":
                        adjacent_cell_keys.add(e.dst[1])

                if not adjacent_cell_keys:
                    continue

                all_orphan = True
                for ck in adjacent_cell_keys:
                    if ck not in self.cells or self.cells[ck].keep:
                        all_orphan = False
                        break
                    cell_degree = sum(
                        1 for e in self.edges
                        if e.src == ("cell", ck) or e.dst == ("cell", ck)
                    )
                    if cell_degree != 1:
                        all_orphan = False
                        break

                if all_orphan:
                    for ck in adjacent_cell_keys:
                        self.cells.pop(ck, None)
                    for e in op_in_edges + op_out_edges:
                        self.edges.discard(e)
                    del self.ops[op_key]
                    changed = True
                    break

    def annotate_runtime(self) -> None:
        """Classify current cell contents and attach annotations for enriched rendering.

        Must be called *after* the network has settled (i.e. after ``run()``).
        All cells recorded during capture have their ``.content`` inspected and a
        :class:`CellAnnotation` stored on the corresponding :class:`CellNode`.

        Returns the :data:`PremiseColorMap` (``id(premise) \u2192 colour``) that was
        used, which is also stored as ``self._premise_colors`` for the DOT renderers.
        """
        try:
            from .supported_values import Supported as _Supported, _unwrap_support as _unwrap
            from .tms import Tms as _Tms, tms_query as _tms_query, premise_in as _premise_in
        except ImportError:
            return {}

        # Collect all premises that appear in the currently-believed support across
        # all tracked cells, preserving first-encounter order for stable colouring.
        all_premises: List[Any] = []
        seen_ids: Set[int] = set()

        def _gather(content: Any) -> None:
            def _add(p: Any) -> None:
                if id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    all_premises.append(p)

            if isinstance(content, _Tms):
                believed = _tms_query(content)
                if isinstance(believed, _Supported):
                    for p in _unwrap(believed.support):
                        _add(p)
            elif isinstance(content, _Supported):
                for p in _unwrap(content.support):
                    _add(p)

        for cell_obj in self._cell_objs.values():
            _gather(getattr(cell_obj, "content", None))

        premise_colors = _build_premise_color_map(all_premises)

        # Build human-readable labels for the legend.
        premise_labels: Dict[int, str] = {}
        for p in all_premises:
            desc = p.describe() if hasattr(p, "describe") else repr(p)
            if len(desc) > 30:
                desc = desc[:27] + "\u2026"
            premise_labels[id(p)] = desc.replace('"', "'")

        # Attach annotation to every CellNode we still have a live object for.
        for cell_key, cell_obj in self._cell_objs.items():
            if cell_key not in self.cells:
                continue
            content = getattr(cell_obj, "content", None)
            self.cells[cell_key].annotation = _classify_content(content, premise_colors, cell_obj)

        self._premise_colors = premise_colors
        self._premise_color_labels = premise_labels

    @property
    def premise_colors(self) -> PremiseColorMap:
        """Premise-to-colour mapping built by :meth:`annotate_runtime`. Do not mutate."""
        return self._premise_colors

    def to_dot(self, *, rankdir: str = "LR") -> str:
        return self._to_structural_dot(rankdir=rankdir)

    def _effective_edges(self) -> Set[Tuple[Tuple[str, int], Tuple[str, int]]]:
        """Return compact flow edges, bypassing anonymous cells."""
        keep_cells = {k for k, c in self.cells.items() if c.keep}
        keep_cells.update(self._param_cell_tags.keys())

        outgoing: Dict[int, List[Tuple[str, int]]] = {}
        for edge in self.edges:
            if edge.src[0] == "cell":
                outgoing.setdefault(edge.src[1], []).append(edge.dst)

        effective_edges: Set[Tuple[Tuple[str, int], Tuple[str, int]]] = set()

        for edge in self.edges:
            src, dst = edge.src, edge.dst
            if src[0] != "op":
                continue

            if dst[0] == "op":
                effective_edges.add((src, dst))
                continue

            if dst[0] == "cell" and dst[1] in keep_cells:
                effective_edges.add((src, dst))
                continue

            if dst[0] == "cell":
                for next_dst in outgoing.get(dst[1], []):
                    if next_dst[0] == "op":
                        effective_edges.add((src, next_dst))

        for edge in self.edges:
            src, dst = edge.src, edge.dst
            if src[0] == "cell" and src[1] in keep_cells and dst[0] == "op":
                effective_edges.add((src, dst))

        return effective_edges

    def to_hierarchical_dot(self, *, rankdir: str = "LR") -> str:
        """Render merged operation flow clustered by parent function.
        
        Key semantics:
        - Thick lines: cell used at same hierarchical level where it was declared
        - Dotted lines: cell passed down as parameter into nested scopes
        - Functions consolidated by name at each hierarchy level (not invocation tokens)
        """
        flow_counts: Dict[Tuple[Tuple[str, str, str], Tuple[str, str, str]], int] = {}
        effective_edges = self._effective_edges()

        cell_defs: Dict[str, Dict[str, Any]] = {}
        op_groups: Dict[str, Dict[str, str]] = {}
        op_hierarchy: Dict[Tuple[str, ...], Dict[str, str]] = {}

        def group_token_for_op(op_key: int) -> str:
            op = self.ops[op_key]
            return op.group_path[-1] if op.group_path else "global"

        def full_group_path(op_key: int) -> Tuple[str, ...]:
            op = self.ops[op_key]
            return op.group_path if op.group_path else ("global",)

        def canonical_group_name(group_token: str) -> str:
            info = self._group_instances.get(group_token)
            if info is None:
                return group_token
            return info.name

        def canonical_hierarchy_path(group_path: Tuple[str, ...]) -> Tuple[str, ...]:
            """Convert invocation tokens to canonical function names."""
            return tuple(canonical_group_name(token) for token in group_path)

        def group_label(group_token: str) -> str:
            return canonical_group_name(group_token)

        def canonical_param_tags(cell_key: int) -> List[Tuple[str, str]]:
            tags = {
                (canonical_group_name(group_token), param_name)
                for group_token, param_name in self._param_cell_tags.get(cell_key, set())
            }
            return sorted(tags)

        def simple_cell_role(cell_key: int, group_name: str) -> str:
            for tagged_group, param_name in canonical_param_tags(cell_key):
                if tagged_group == group_name:
                    return param_name

            label = self.cells[cell_key].label
            if label != "cell":
                return label

            other_params = sorted({param_name for _, param_name in canonical_param_tags(cell_key)})
            if other_params:
                return "/".join(other_params)
            return ""

        def cell_token(cell_key: int) -> Tuple[str, str]:
            tags = canonical_param_tags(cell_key)
            tag_names = sorted({param_name for _, param_name in tags})
            base_label = self.cells[cell_key].label

            if tag_names:
                # Merge parameter cells by conceptual role, not object identity.
                label = tag_names[0] if len(tag_names) == 1 else "/".join(tag_names)
                token = f"param_shared__{_slug(label)}"
            else:
                token = f"cell_{cell_key}"
                label = base_label

            group_ports: Dict[Tuple[str, str], str] = {}
            for group_name, param_name in sorted(tags):
                port_key = (group_name, param_name)
                if port_key not in group_ports:
                    group_ports[port_key] = f"{_slug(group_name)}__{_slug(param_name) or 'value'}"

            port_counts: Dict[str, int] = {}
            for _, param_name in sorted(tags):
                port_counts[param_name] = port_counts.get(param_name, 0) + 1

            port_entries: List[Tuple[str, str]] = []
            for group_name, param_name in sorted(tags):
                port_id = group_ports[(group_name, param_name)]
                if port_counts[param_name] > 1:
                    port_text = f"{group_label(group_name)}.{param_name}"
                else:
                    port_text = param_name
                port_entries.append((port_id, port_text))

            cell_defs[token] = {
                "label": label,
                "is_param": bool(tags),
                "shared": len({group for group, _ in tags}) > 1,
                "ports": port_entries,
                "group_ports": group_ports,
                "annotation": self.cells[cell_key].annotation if cell_key in self.cells else None,
            }
            return token, label

        def describe_cell_for_group(cell_key: int, group_token: str) -> str:
            group_name = canonical_group_name(group_token)
            simple_role = simple_cell_role(cell_key, group_name)
            if simple_role:
                return simple_role

            producer_edges = sorted(
                [
                    edge
                    for edge in self.edges
                    if edge.dst == ("cell", cell_key)
                    and edge.src[0] == "op"
                    and canonical_group_name(group_token_for_op(edge.src[1])) == group_name
                ],
                key=lambda edge: (self.ops[edge.src[1]].name, edge.label, edge.src[1]),
            )
            if producer_edges:
                producer = producer_edges[0]
                producer_name = self.ops[producer.src[1]].name
                return f"{producer_name}.{producer.label}"

            consumer_edges = sorted(
                [
                    edge
                    for edge in self.edges
                    if edge.src == ("cell", cell_key)
                    and edge.dst[0] == "op"
                    and canonical_group_name(group_token_for_op(edge.dst[1])) == group_name
                ],
                key=lambda edge: (self.ops[edge.dst[1]].name, edge.label, edge.dst[1]),
            )
            if consumer_edges:
                consumer = consumer_edges[0]
                consumer_name = self.ops[consumer.dst[1]].name
                return f"{consumer_name}.{consumer.label}"

            return f"cell{cell_key}"

        op_label_counts: Dict[Tuple[str, str], int] = {}
        for op_key, op in self.ops.items():
            group_name = canonical_group_name(group_token_for_op(op_key))
            op_label_counts[(group_name, op.label)] = op_label_counts.get((group_name, op.label), 0) + 1

        def op_signature_parts(op_key: int, group_name: str) -> List[str]:
            parts: List[str] = []
            for edge in sorted(self.edges, key=lambda item: (item.label, item.src, item.dst)):
                if edge.dst == ("op", op_key) and edge.src[0] == "cell":
                    role = simple_cell_role(edge.src[1], group_name)
                    if role:
                        parts.append(f"{edge.label}={role}")
                elif edge.src == ("op", op_key) and edge.dst[0] == "cell":
                    role = simple_cell_role(edge.dst[1], group_name)
                    if role:
                        parts.append(f"{edge.label}={role}")
            return parts

        def op_site_token(op_key: int) -> str:
            op = self.ops[op_key]
            group_name = canonical_group_name(group_token_for_op(op_key))
            parts = op_signature_parts(op_key, group_name)
            if not parts and op_label_counts[(group_name, op.label)] == 1:
                return f"{group_name}::{op.label}"
            signature = ", ".join(parts)
            return f"{group_name}::{op.label}[{signature}]"

        def op_display_label(op_key: int) -> str:
            op = self.ops[op_key]
            group_name = canonical_group_name(group_token_for_op(op_key))
            parts = op_signature_parts(op_key, group_name)
            if not parts and op_label_counts[(group_name, op.label)] == 1:
                return op.label

            signature = ", ".join(parts)
            return f"{op.label}[{signature}]"

        def node_group(ref: Tuple[str, int], *, peer_group: Optional[str] = None) -> Tuple[str, str, str]:
            kind, key = ref
            if kind == "op":
                group_name = canonical_group_name(group_token_for_op(key))
                op_token = op_site_token(key)
                op_id = f"op_{_slug(op_token)}"
                # Use canonical hierarchy (function names, not invocation tokens)
                full_path = full_group_path(key)
                canonical_path = canonical_hierarchy_path(full_path)
                op_hierarchy.setdefault(canonical_path, {})[op_id] = op_display_label(key)
                op_groups.setdefault(group_name, {})[op_id] = op_display_label(key)
                return ("op", op_id, "")

            token, _ = cell_token(key)
            port = ""
            if peer_group:
                port_role = simple_cell_role(key, peer_group)
                port = cell_defs[token]["group_ports"].get((peer_group, port_role), "")
            return ("cell", token, port)

        for src, dst in effective_edges:
            src_peer_group = canonical_group_name(group_token_for_op(dst[1])) if dst[0] == "op" else None
            dst_peer_group = canonical_group_name(group_token_for_op(src[1])) if src[0] == "op" else None
            gsrc = node_group(src, peer_group=src_peer_group)
            gdst = node_group(dst, peer_group=dst_peer_group)
            flow_counts[(gsrc, gdst)] = flow_counts.get((gsrc, gdst), 0) + 1

        _key_table = (
            '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4" CELLPADDING="0" BGCOLOR="white">'
            '<TR><TD COLSPAN="2" BGCOLOR="#e0e0e0" ALIGN="LEFT" CELLPADDING="3"><B>cells</B></TD></TR>'
            '<TR>'
              '<TD BGCOLOR="#ffffff" BORDER="1" STYLE="rounded" CELLPADDING="4"> cell </TD>'
              '<TD ALIGN="LEFT" CELLPADDING="4"> no value yet</TD>'
            '</TR>'
            '<TR>'
              '<TD BGCOLOR="#f0f0f0" BORDER="1" STYLE="rounded" CELLPADDING="4"> 3.14 </TD>'
              '<TD ALIGN="LEFT" CELLPADDING="4"> plain value</TD>'
            '</TR>'
            '<TR>'
              '<TD BGCOLOR="#e8f4fd" BORDER="1" STYLE="rounded" CELLPADDING="4"> [a, b] </TD>'
              '<TD ALIGN="LEFT" CELLPADDING="4"> interval value</TD>'
            '</TR>'
            '<TR>'
              '<TD BGCOLOR="#4e79a7" BORDER="2" STYLE="rounded" CELLPADDING="4">'
                '<FONT COLOR="white"><B>5</B></FONT>'
                '<FONT COLOR="#cccccc" POINT-SIZE="8"> 2 4…</FONT>'
              '</TD>'
              '<TD ALIGN="LEFT" CELLPADDING="4"> TMS: <B>bold</B>=believed, grey=alternatives</TD>'
            '</TR>'
            '<TR>'
              '<TD BGCOLOR="#fff6d5" BORDER="1" STYLE="rounded" CELLPADDING="4"> x </TD>'
              '<TD ALIGN="LEFT" CELLPADDING="4"> shared input / output cell</TD>'
            '</TR>'
            '<TR><TD COLSPAN="2" BGCOLOR="#e0e0e0" ALIGN="LEFT" CELLPADDING="3"><B>propagators</B></TD></TR>'
            '<TR>'
              '<TD BGCOLOR="#f2f2f2" BORDER="1" STYLE="rounded" CELLPADDING="4"> adder </TD>'
              '<TD ALIGN="LEFT" CELLPADDING="4"> propagator — applies a constraint</TD>'
            '</TR>'
            '<TR><TD COLSPAN="2" BGCOLOR="#e0e0e0" ALIGN="LEFT" CELLPADDING="3"><B>edges</B></TD></TR>'
            '<TR>'
              '<TD BGCOLOR="#dde4f0" BORDER="1" STYLE="rounded" CELLPADDING="4">'
                '<FONT COLOR="#5b6c8f">━━▶</FONT>'
              '</TD>'
              '<TD ALIGN="LEFT" CELLPADDING="4"><FONT COLOR="#5b6c8f"> cell read by propagator</FONT></TD>'
            '</TR>'
            '<TR>'
              '<TD BGCOLOR="#ddf0e4" BORDER="1" STYLE="rounded" CELLPADDING="4">'
                '<FONT COLOR="#2f7d4a">━━▶</FONT>'
              '</TD>'
              '<TD ALIGN="LEFT" CELLPADDING="4"><FONT COLOR="#2f7d4a"> cell written by propagator</FONT></TD>'
            '</TR>'
            '</TABLE>>'
        )

        lines: List[str] = [
            "digraph PropagatorHierarchy {",
            f"  rankdir={rankdir};",
            "  graph [fontsize=10, fontname=\"Helvetica\", splines=spline, overlap=false,"
            "         nodesep=0.3, ranksep=0.7, newrank=true, margin=0.2];",
            "  node [fontname=\"Helvetica\", fontsize=9];",
            "  edge [fontname=\"Helvetica\", fontsize=8];",
            f"  key_table [shape=plain, label={_key_table}] ;",
            "  subgraph cluster_key {",
            "    label=\"symbol key\";",
            "    color=\"#d8d8d8\";",
            "    style=\"rounded,dashed\";",
            "    fontsize=10;",
            "    margin=8;",
            "    key_table;",
            "  }",
        ]

        for token in sorted(cell_defs):
            label = cell_defs[token]["label"]
            is_param = cell_defs[token]["is_param"]
            shared = cell_defs[token]["shared"]
            ports = cell_defs[token]["ports"]
            if ports:
                port_cells = "".join(
                    f'<TD PORT="{port_id}" BGCOLOR="#fffdf1">{port_label}</TD>' for port_id, port_label in ports
                )
                outer_attrs = 'BORDER="1" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4"'
                header_color = "#fff6d5"
                if shared:
                    outer_attrs = 'BORDER="1.4" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4"'
                ann = cell_defs[token].get("annotation")
                if ann and ann.content_type in ("tms", "supported", "interval", "plain") and ann.display_value:
                    fill = ann.fill_color or header_color
                    header_text = f"<B>{label}</B>"
                    if ann.content_type == "tms" and ann.tms_branches:
                        believed = [v for v, b in ann.tms_branches if b]
                        others = [v for v, b in ann.tms_branches if not b]
                        bv = believed[0] if believed else "?"
                        ov = ", ".join(others[:3]) + ("…" if len(others) > 3 else "")
                        content_row = (
                            f'<TR><TD BGCOLOR="{fill}" COLSPAN="{len(ports)}">'
                            f'<B>{bv}</B>'
                            + (f' <FONT POINT-SIZE="8" COLOR="#eeeeee">{ov}</FONT>' if ov else "")
                            + "</TD></TR>"
                        )
                        border_width = "2"
                    else:
                        content_row = (
                            f'<TR><TD BGCOLOR="{fill}" COLSPAN="{len(ports)}">'
                            f'<FONT POINT-SIZE="9">{ann.display_value}</FONT>'
                            f"</TD></TR>"
                        )
                        border_width = "1"
                    # Override border colour for annotated cells
                    outer_attrs = outer_attrs.replace('BORDER="1"', f'BORDER="{border_width}"').replace('BORDER="1.4"', f'BORDER="{border_width}"')
                    html_label = (
                        f"<<TABLE {outer_attrs}>"
                        f'<TR><TD BGCOLOR="{header_color}" COLSPAN="{len(ports)}">{header_text}</TD></TR>'
                        f"{content_row}"
                        f"<TR>{port_cells}</TR>"
                        "</TABLE>>"
                    )
                else:
                    html_label = (
                        f'<<TABLE {outer_attrs}>'
                        f'<TR><TD BGCOLOR="{header_color}" COLSPAN="{len(ports)}">{label}</TD></TR>'
                        f'<TR>{port_cells}</TR>'
                        "</TABLE>>"
                    )
                lines.append(f"  {token} [shape=plain, label={html_label}];")
            else:
                ann = cell_defs[token].get("annotation")
                lbl, style = _annotation_dot_attrs(label, ann, is_param=is_param, shared=shared)
                lines.append(f"  {token} [shape=ellipse, {style}, label={lbl}] ;")

        def render_nested_clusters(path: Tuple[str, ...], indent: int = 1) -> List[str]:
            """Recursively render nested cluster hierarchy.
            
            Functions are consolidated by name at each level, so multiple calls
            to the same function at the same scope appear in one cluster.
            """
            cluster_lines: List[str] = []
            indent_str = "  " * indent
            
            # Check if this is a recursive call (same function name appearing twice in path)
            func_names = list(path)
            is_recursive = len(func_names) != len(set(func_names))
            
            cluster_id = f"cluster_{'_'.join(_slug(p) for p in path)}"
            label_text = path[-1] if path else "global"
            
            if is_recursive:
                # Mark recursive cluster with special indicator
                label_text = f"{path[-1]} (recursive)"
            
            cluster_lines.append(f"{indent_str}subgraph {cluster_id} {{")
            cluster_lines.append(f"{indent_str}  label=\"{label_text}\";")
            cluster_lines.append(f"{indent_str}  color=\"#bcbcbc\";")
            cluster_lines.append(f"{indent_str}  style=\"rounded\";")
            cluster_lines.append(f"{indent_str}  margin=10;")
            
            if path in op_hierarchy:
                for op_id, op_label in sorted(op_hierarchy[path].items()):
                    cluster_lines.append(
                        f"{indent_str}  {op_id} [shape=box, style=\"rounded,filled\", fillcolor=\"#f2f2f2\", label=\"{op_label}\"] ;"
                    )
            
            # Only add child clusters if this is NOT a recursive instance
            if not is_recursive:
                child_paths = sorted({p for p in op_hierarchy if len(p) == len(path) + 1 and p[:len(path)] == path})
                for child_path in child_paths:
                    cluster_lines.extend(render_nested_clusters(child_path, indent + 1))
            
            cluster_lines.append(f"{indent_str}}}")
            return cluster_lines
        
        root_paths = sorted({p for p in op_hierarchy if len(p) == 1})
        for root_path in root_paths:
            lines.extend(render_nested_clusters(root_path))

        for (src, dst), count in sorted(
            flow_counts.items(),
            key=lambda item: (item[0][0][0], item[0][0][1], item[0][0][2], item[0][1][0], item[0][1][1], item[0][1][2]),
        ):
            if src[0] == "cell":
                src_id = src[1] if not src[2] else f'{src[1]}:{src[2]}'
            else:
                src_id = src[1]

            if dst[0] == "cell":
                dst_id = dst[1] if not dst[2] else f'{dst[1]}:{dst[2]}'
            else:
                dst_id = dst[1]

            edge_label = f"x{count}" if count > 1 else ""
            
            if src[0] == "cell" and dst[0] == "op":
                # Cell being READ by operation
                style_parts = ['color="#5b6c8f"']
                if edge_label:
                    style_parts.append(f'label="{edge_label}"')
                attrs = '[' + ', '.join(style_parts) + ']'
                lines.append(f"  {src_id} -> {dst_id} {attrs};")
            elif src[0] == "op" and dst[0] == "cell":
                # Cell being WRITTEN by operation
                style_parts = ['color="#2f7d4a"']
                if edge_label:
                    style_parts.append(f'label="{edge_label}"')
                attrs = '[' + ', '.join(style_parts) + ']'
                lines.append(f"  {src_id} -> {dst_id} {attrs};")
            else:
                if edge_label:
                    lines.append(f"  {src_id} -> {dst_id} [label=\"{edge_label}\"];")
                else:
                    lines.append(f"  {src_id} -> {dst_id};")

        # \u2500\u2500 Premise colour legend (only when annotate_runtime() was called) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if self._premise_colors:
            _light_fills = {"#edc948", "#bab0ac", "#ff9da7"}
            # Single HTML-table node avoids all rank-constraint / cluster conflicts
            ncols = 2
            items = list(self._premise_colors.items())
            trows = [
                '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="3" CELLPADDING="0" BGCOLOR="white">',
                '<TR><TD COLSPAN="{}" BGCOLOR="#d0d0d0" ALIGN="LEFT" CELLPADDING="3">'
                '<B>premise colours</B></TD></TR>'.format(ncols),
            ]
            for row_start in range(0, len(items), ncols):
                chunk = items[row_start:row_start + ncols]
                cells = ""
                for pid, color in chunk:
                    plabel = self._premise_color_labels.get(pid, f"premise {row_start}")
                    plabel_html = (
                        plabel.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    )
                    fontcolor = "#333333" if color in _light_fills else "white"
                    cells += (
                        f'<TD BGCOLOR="{color}" CELLPADDING="4" BORDER="1" STYLE="rounded">'
                        f'<FONT COLOR="{fontcolor}" POINT-SIZE="9">{plabel_html}</FONT></TD>'
                    )
                if len(chunk) < ncols:
                    cells += "<TD></TD>" * (ncols - len(chunk))
                trows.append(f"<TR>{cells}</TR>")
            trows.append("</TABLE>")
            _legend_table = "<" + "".join(trows) + ">"
            lines.append(f"  premise_legend_table [shape=plain, label={_legend_table}] ;")
            lines.append("  subgraph cluster_premise_legend {")
            lines.append('    label="";')
            lines.append('    color="#d0d0d0";')
            lines.append('    style="rounded,dashed";')
            lines.append("    premise_legend_table;")
            lines.append("  }")
        lines.append("}")

        return "\n".join(lines)

    def _to_structural_dot(self, *, rankdir: str = "LR") -> str:
        lines: List[str] = [
            "digraph PropagatorCircuit {",
            f"  rankdir={rankdir};",
            "  graph [fontsize=11, fontname=\"Helvetica\", splines=true, overlap=false];",
            "  node [fontname=\"Helvetica\", fontsize=10];",
            "  edge [fontname=\"Helvetica\", fontsize=9];",
        ]

        for key, cell in sorted(self.cells.items()):
            if not cell.keep:
                lines.append(f"  cell_{key} [shape=point, label=\"\"];")
                continue
            lbl, style = _annotation_dot_attrs(cell.label, cell.annotation)
            lines.append(f"  cell_{key} [shape=ellipse, {style}, label={lbl}];")

        for key, op in sorted(self.ops.items()):
            lines.append(
                f"  op_{key} [shape=box, style=\"rounded,filled\", fillcolor=\"#f2f2f2\", label=\"{op.label}\"] ;"
            )

        for edge in sorted(self.edges, key=lambda e: (e.src, e.dst, e.label)):
            src = f"{edge.src[0]}_{edge.src[1]}"
            dst = f"{edge.dst[0]}_{edge.dst[1]}"
            if edge.label:
                lines.append(f"  {src} -> {dst} [label=\"{edge.label}\"];")
            else:
                lines.append(f"  {src} -> {dst};")

        lines.append("}")
        return "\n".join(lines)

    def to_behavioral_dot(self, *, rankdir: str = "LR") -> str:
        """Return a highly merged behavior graph.

        This view merges all anonymous cells away and groups operations by label,
        so recursive or iterative unrolling is summarized as weighted flow edges.
        """
        flow_counts: Dict[Tuple[Tuple[str, str], Tuple[str, str]], int] = {}

        effective_edges = self._effective_edges()

        def node_group(ref: Tuple[str, int]) -> Tuple[str, str]:
            kind, key = ref
            if kind == "op":
                return ("op", self.ops[key].label)
            return ("cell", self.cells[key].label)

        for src, dst in effective_edges:
            gsrc = node_group(src)
            gdst = node_group(dst)
            flow_counts[(gsrc, gdst)] = flow_counts.get((gsrc, gdst), 0) + 1

        op_labels = sorted({grp[1] for pair in flow_counts for grp in pair if grp[0] == "op"})
        cell_labels = sorted({grp[1] for pair in flow_counts for grp in pair if grp[0] == "cell"})

        lines: List[str] = [
            "digraph PropagatorBehavior {",
            f"  rankdir={rankdir};",
            "  graph [fontsize=11, fontname=\"Helvetica\", splines=true, overlap=false];",
            "  node [fontname=\"Helvetica\", fontsize=10];",
            "  edge [fontname=\"Helvetica\", fontsize=9];",
        ]

        for label in cell_labels:
            lines.append(f"  cell_{_slug(label)} [shape=ellipse, label=\"{label}\"];")
        for label in op_labels:
            lines.append(
                f"  op_{_slug(label)} [shape=box, style=\"rounded,filled\", fillcolor=\"#f2f2f2\", label=\"{label}\"] ;"
            )

        for (src, dst), count in sorted(flow_counts.items(), key=lambda item: (item[0][0], item[0][1], item[1])):
            src_id = f"{src[0]}_{_slug(src[1])}"
            dst_id = f"{dst[0]}_{_slug(dst[1])}"
            label = f"x{count}" if count > 1 else ""
            if label:
                lines.append(f"  {src_id} -> {dst_id} [label=\"{label}\"];")
            else:
                lines.append(f"  {src_id} -> {dst_id};")

        lines.append("}")
        return "\n".join(lines)

    def write_dot(
        self,
        path: Union[str, Path],
        *,
        compact: bool = True,
        mode: str = "structural",
    ) -> Path:
        out_path = Path(path)
        if compact:
            self.compact()
        if mode == "behavioral":
            dot_text = self.to_behavioral_dot()
        elif mode == "hierarchical":
            dot_text = self.to_hierarchical_dot()
        else:
            dot_text = self.to_dot()
        out_path.write_text(dot_text, encoding="utf-8")
        return out_path

    def render(
        self,
        path: Union[str, Path],
        *,
        mode: str = "behavioral",
        image_format: str = "png",
        compact: bool = True,
    ) -> Path:
        """Render with python-graphviz if available.

        Requires: pip install graphviz and Graphviz CLI on PATH.
        """
        if compact:
            self.compact()

        if mode == "behavioral":
            dot_text = self.to_behavioral_dot()
        elif mode == "hierarchical":
            dot_text = self.to_hierarchical_dot()
        else:
            dot_text = self.to_dot()

        try:
            import graphviz
        except ImportError as exc:
            raise RuntimeError(
                "python-graphviz is not installed. Install with 'pip install graphviz'."
            ) from exc

        out_path = Path(path)
        source = graphviz.Source(dot_text)
        rendered = source.render(
            filename=out_path.stem,
            directory=str(out_path.parent),
            format=image_format,
            cleanup=True,
        )
        return Path(rendered)


@dataclass
class _GroupInstance:
    """Canonical name and display label for a function invocation group."""
    name: str
    label: str


class TraceSpec(NamedTuple):
    """Wiring specification for a single propagator primitive.

    Positional unpacking is supported: ``in_ix, out_ix, label_fn = spec``.
    """
    input_indexes: Tuple[int, ...]
    output_indexes: Tuple[int, ...]
    label_fn: Optional[Callable[..., str]]


# Maps primitive function names to their wiring specification.
# Call register_trace_spec() to add entries for your own primitives.
_TRACE_SPECS: Dict[str, TraceSpec] = {
    # Arithmetic (unidirectional)
    "adder":          TraceSpec((0, 1), (2,), None),
    "subtractor":     TraceSpec((0, 1), (2,), None),
    "multiplier":     TraceSpec((0, 1), (2,), None),
    "divider":        TraceSpec((0, 1), (2,), None),
    "squarer":        TraceSpec((0,),   (1,), None),
    "sqrter":         TraceSpec((0,),   (1,), None),
    "absolute_value": TraceSpec((0,),   (1,), None),
    "abs_value":      TraceSpec((0,),   (1,), None),
    "absoluter":      TraceSpec((0,),   (1,), None),
    # Comparison (unidirectional)
    "eq":             TraceSpec((0, 1), (2,), None),
    "lt":             TraceSpec((0, 1), (2,), None),
    "gt":             TraceSpec((0, 1), (2,), None),
    "lte":            TraceSpec((0, 1), (2,), None),
    "gte":            TraceSpec((0, 1), (2,), None),
    # Aliases for comparison operators
    "equal_to":       TraceSpec((0, 1), (2,), None),
    "less_than":      TraceSpec((0, 1), (2,), None),
    "greater_than":   TraceSpec((0, 1), (2,), None),
    # Boolean (unidirectional)
    "inverter":       TraceSpec((0,),   (1,), None),
    "conjoiner":      TraceSpec((0, 1), (2,), None),
    "disjoiner":      TraceSpec((0, 1), (2,), None),
    # Aliases for boolean operators
    "neg":            TraceSpec((0,),   (1,), None),
    "negate":         TraceSpec((0,),   (1,), None),
    "and_gate":       TraceSpec((0, 1), (2,), None),
    "or_gate":        TraceSpec((0, 1), (2,), None),
    # Control flow
    "switch":         TraceSpec((0, 1), (2,), None),
    "conditional":    TraceSpec((0, 1, 2), (3,), None),
    # Value injection
    "constant":       TraceSpec((),     (1,), lambda value, *_: f"const({value})"),
    # Guessing / TMS constraint helpers
    # require(cell): writes True into cell — the cell is an output
    "require":        TraceSpec((),     (0,), lambda cell: "require"),
    # abhor(cell): writes False into cell — the cell is an output
    "abhor":          TraceSpec((),     (0,), lambda cell: "abhor"),
    # one_of(values, output_cell): creates a choice tree rooted at output_cell
    "one_of":         TraceSpec((),     (1,), lambda values, cell: f"one_of({list(values)})"),
    # binary_amb(cell, ...): writes a TMS amb value into cell
    "binary_amb":     TraceSpec((),     (0,), lambda cell, *_: "amb"),
}


def register_trace_spec(
    name: str,
    input_indexes: Tuple[int, ...],
    output_indexes: Tuple[int, ...],
    label_fn: Optional[Callable[..., str]] = None,
) -> None:
    """Register a custom propagator constructor so it is captured during tracing.

    Call this once at module level, before any ``capture_circuit`` context is
    entered, to make your own primitives visible in circuit diagrams.

    Args:
        name: Exact name of the function as it appears in the target namespace.
        input_indexes: Positional-argument indexes that are input cells.
        output_indexes: Positional-argument indexes that are output cells.
        label_fn: Optional callable ``(*args) -> str`` for a custom node label.
            Defaults to using *name* as the label.

    Example::

        from propagator.circuit_viz import register_trace_spec

        # my_gate(in_a, in_b, out) — two inputs, one output
        register_trace_spec("my_gate", input_indexes=(0, 1), output_indexes=(2,))
    """
    _TRACE_SPECS[name] = TraceSpec(input_indexes, output_indexes, label_fn)


class CircuitCapture:
    """Context manager that patches propagator constructors to trace wiring."""

    def __init__(
        self,
        *,
        trace: Optional[CircuitTrace] = None,
        target_modules: Optional[Sequence[ModuleType]] = None,
        target_namespaces: Optional[Sequence[Dict[str, Any]]] = None,
        pin_cells: Optional[Iterable[Any]] = None,
        group_function_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.trace = trace or CircuitTrace()
        self.target_modules = list(target_modules or [])
        self.target_namespaces = list(target_namespaces or [])
        self.group_function_names = list(group_function_names or [])
        self._patches: List[Tuple[Any, str, Any]] = []
        self._group_stack: List[str] = []
        self._group_seq = 0
        if pin_cells:
            self.trace.pin_cells(pin_cells)

    def _wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        in_ix, out_ix, label_builder = _TRACE_SPECS[name]

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            inputs = [args[i] for i in in_ix]
            outputs = [args[i] for i in out_ix]
            label = label_builder(*args) if label_builder is not None else name
            self.trace.add_operation(
                name=name,
                inputs=inputs,
                outputs=outputs,
                label=label,
                group_path=tuple(self._group_stack),
            )
            return result

        wrapped.__name__ = getattr(fn, "__name__", name)
        wrapped.__doc__ = getattr(fn, "__doc__", None)
        return wrapped

    def _patch_object_attr(self, obj: Any, attr: str) -> None:
        if not hasattr(obj, attr):
            return
        original = getattr(obj, attr)
        wrapped = self._wrap(attr, original)
        setattr(obj, attr, wrapped)
        self._patches.append((obj, attr, original))

    def _patch_namespace(self, ns: Dict[str, Any], attr: str) -> None:
        if attr not in ns:
            return
        original = ns[attr]
        wrapped = self._wrap(attr, original)
        ns[attr] = wrapped
        self._patches.append((ns, attr, original))

    def _wrap_group_function(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            signature = None

        def bind_arguments(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[inspect.BoundArguments]:
            if signature is None:
                return None

            try:
                return signature.bind_partial(*args, **kwargs)
            except TypeError:
                return None

        def describe_value(value: Any) -> str:
            cells = self._extract_cells(value)
            if len(cells) == 1:
                cell_key = self.trace._cell_key(cells[0])
                cell_label = self.trace.cells[cell_key].label
                if cell_label != "cell":
                    return cell_label
                return f"cell{cell_key}"
            if len(cells) > 1:
                return "+".join(describe_value(cell) for cell in cells)
            return repr(value)

        def group_instance(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[str, str]:
            self._group_seq += 1
            token = f"{name}__{self._group_seq}"
            bound = bind_arguments(args, kwargs)
            if bound is None or not bound.arguments:
                return token, f"{name} #{self._group_seq}"

            parts = [describe_value(value) for value in bound.arguments.values()]
            return token, f"{name}({', '.join(parts)})"

        def register_params(group_token: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
            bound = bind_arguments(args, kwargs)
            if bound is None:
                return

            for param_name, value in bound.arguments.items():
                for cell in self._extract_cells(value):
                    self.trace.register_param_cell(cell, group_token, param_name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            group_token, group_display = group_instance(args, kwargs)
            self.trace.register_group_instance(group_token, name, group_display)
            register_params(group_token, args, kwargs)
            self._group_stack.append(group_token)
            try:
                return fn(*args, **kwargs)
            finally:
                self._group_stack.pop()

        wrapped.__name__ = getattr(fn, "__name__", name)
        wrapped.__doc__ = getattr(fn, "__doc__", None)
        return wrapped

    def _extract_cells(self, value: Any) -> List[Any]:
        """Extract cell-like objects from function arguments."""
        if self._is_cell_like(value):
            return [value]

        if isinstance(value, (list, tuple)):
            cells: List[Any] = []
            for item in value:
                cells.extend(self._extract_cells(item))
            return cells

        return []

    def _is_cell_like(self, value: Any) -> bool:
        return hasattr(value, "neighbors") and hasattr(value, "add_content")

    def _patch_namespace_group_function(self, ns: Dict[str, Any], attr: str) -> None:
        if attr not in ns:
            return
        original = ns[attr]
        wrapped = self._wrap_group_function(attr, original)
        ns[attr] = wrapped
        self._patches.append((ns, attr, original))

    def _wrap_compound(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(neighbors: Any, to_build: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            captured_group = tuple(self._group_stack)

            if callable(to_build) and captured_group:
                def grouped_to_build(*inner_args: Any, **inner_kwargs: Any) -> Any:
                    self._group_stack.extend(captured_group)
                    try:
                        return to_build(*inner_args, **inner_kwargs)
                    finally:
                        del self._group_stack[-len(captured_group):]
            else:
                grouped_to_build = to_build

            return fn(neighbors, grouped_to_build, *args, **kwargs)

        wrapped.__name__ = getattr(fn, "__name__", "compound_propagator")
        wrapped.__doc__ = getattr(fn, "__doc__", None)
        return wrapped

    def _patch_namespace_compound(self, ns: Dict[str, Any]) -> None:
        attr = "compound_propagator"
        if attr not in ns:
            return
        original = ns[attr]
        wrapped = self._wrap_compound(original)
        ns[attr] = wrapped
        self._patches.append((ns, attr, original))

    def __enter__(self) -> CircuitTrace:
        for mod in self.target_modules:
            for name in _TRACE_SPECS:
                self._patch_object_attr(mod, name)

        for ns in self.target_namespaces:
            for name in _TRACE_SPECS:
                self._patch_namespace(ns, name)

        for ns in self.target_namespaces:
            for name in self.group_function_names:
                self._patch_namespace_group_function(ns, name)

        for ns in self.target_namespaces:
            self._patch_namespace_compound(ns)

        return self.trace

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        while self._patches:
            owner, attr, original = self._patches.pop()
            if isinstance(owner, dict):
                owner[attr] = original
            else:
                setattr(owner, attr, original)


@contextmanager
def capture_circuit(
    *,
    target_modules: Optional[Sequence[ModuleType]] = None,
    target_namespaces: Optional[Sequence[Dict[str, Any]]] = None,
    pin_cells: Optional[Iterable[Any]] = None,
    group_function_names: Optional[Sequence[str]] = None,
) -> Iterable[CircuitTrace]:
    """Capture a propagator network into a compact circuit trace.

    Typical usage:

        with capture_circuit(target_namespaces=[globals()], pin_cells=[x, answer]) as trace:
            sqrt_network(x, answer)
        trace.write_dot("sqrt_iter.dot")
    """
    cap = CircuitCapture(
        target_modules=target_modules,
        target_namespaces=target_namespaces,
        pin_cells=pin_cells,
        group_function_names=group_function_names,
    )
    try:
        yield cap.__enter__()
    finally:
        cap.__exit__(None, None, None)
