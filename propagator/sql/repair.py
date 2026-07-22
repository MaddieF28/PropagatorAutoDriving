"""Query repair via QR-Hint (Hu et al., SIGMOD 2024) using propagator networks.

Implements the QR-Hint framework for providing actionable, non-leaking hints
to correct wrong SQL queries.  Instead of an external SMT solver (Z3), we
leverage the propagator network's own machinery:

    * **Cells + merge**           → equivalence / contradiction detection
    * **Bidirectional constraints** → backward propagation of expected outputs
    * **TMS + supported values**  → worldview exploration for repair-site search
    * **Hypotheticals + AMB**     → enumerate candidate fixes without Z3

The repair proceeds in QR-Hint's sequential stages:

    FROM  →  WHERE  →  GROUP BY  →  HAVING  →  SELECT

Each stage has a *viability check* expressed as a propagator-network query.
If the check fails, the system computes a minimal set of *repair sites*
(subtrees of the working query's AST that need to change) and generates
graduated hints at the requested level (CLAUSE / CHARACTER / DIRECTION)
without ever revealing the reference query.

Key QR-Hint concepts implemented here
--------------------------------------
* **Table mapping** (§4): bijective alias mapping via column-usage signatures
* **Repair bounds** (§5.1, CreateBounds): computed via backward propagation
  through the *bidirectional* constraint network
* **Repair-site search** (§5, RepairWhere): enumerate candidate site-sets,
  test viability with propagator merge, pick minimum-cost repair
* **Fix derivation** (§5.2, DeriveFixes): push target bounds top-down and
  find smallest formula within each bound
* **FixGrouping** (§6): GROUP BY repair via pair-wise partition equivalence

Also supports legacy mode: diagnose_repair(network, expected_FullRelation)
for comparing against a known expected FullRelation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

from ..cell import Cell, propagator
from ..merge import merge, contradictory_p, the_contradiction
from ..nothing import nothing_p
from .relation_info import (
    ColumnDef, FullRelation, SchemaInfo, EstimateInfo,
    is_relation_info, LatticeLevel,
)
from .network import QueryNetwork


# ═══════════════════════════════════════════════════════════════════════════
# Public data types
# ═══════════════════════════════════════════════════════════════════════════

class HintLevel(IntEnum):
    """Graduated hint disclosure level (QR-Hint §3.1)."""
    CLAUSE    = 1   # Which clause has the error
    CHARACTER = 2   # What kind of error (too restrictive, missing …)
    DIRECTION = 3   # Narrowed guidance (which column, operator type …)


@dataclass
class RepairHint:
    """An actionable hint that does NOT reveal the reference query."""
    clause: str          # SQL clause: "FROM", "WHERE", "GROUP BY", etc.
    severity: str        # "error" | "warning" | "info"
    level: int           # 1, 2, or 3  (HintLevel)
    message: str         # Human-readable hint (never contains reference SQL)
    # ── internal bookkeeping (not shown to user) ──
    _working_fragment: str = ""
    _error_kind: str = ""
    row_impact: Optional[int] = None
    cost: Optional[float] = None      # QR-Hint repair cost (Def 3)
    repair_sites: Optional[list] = None  # AST node indices


@dataclass
class RepairReport:
    """Repair report — never exposes reference SQL."""
    working_sql: str
    hints: list = field(default_factory=list)       # list[RepairHint]
    working_row_count: Optional[int] = None
    reference_row_count: Optional[int] = None       # safe: just a count
    equivalent: bool = False
    stage_results: dict = field(default_factory=dict)  # stage → pass/fail
    # ── internal ──
    _working_output: object = None
    _reference_output: object = None
    _reference_sql: str = ""
    _table_mapping: Optional[dict] = None
    approach_comparison: dict = field(default_factory=dict)


@dataclass
class ApproachRepairResult:
    """Comparable result for one WHERE-repair strategy."""
    approach: str
    repair_sites: list = field(default_factory=list)
    cost: float = 0.0
    viable: bool = False
    details: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Table mapping (QR-Hint §4)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_tables_with_aliases(ast) -> List[Tuple[str, str]]:
    """Return [(table_name, alias), …] from a SELECT AST."""
    tables = []
    from_clause = ast.find(exp.From)
    if from_clause and from_clause.this:
        t = from_clause.this
        tables.append((t.name, t.alias or t.name))
    for j in ast.find_all(exp.Join):
        if j.this:
            tables.append((j.this.name, j.this.alias or j.this.name))
    return tables


def _table_multiset(ast) -> Counter:
    """Tables(Q) as a multiset of table names."""
    return Counter(name for name, _ in _extract_tables_with_aliases(ast))


def _build_alias_signature(ast, alias: str) -> dict:
    """Column-usage signature for one alias (heuristic, QR-Hint §4)."""
    sig: Dict[str, set] = {}
    for col in ast.find_all(exp.Column):
        if col.table == alias:
            sig.setdefault(col.name, set()).add("ref")
    group = ast.find(exp.Group)
    if group:
        for e in group.expressions:
            if isinstance(e, exp.Column) and e.table == alias:
                sig.setdefault(e.name, set()).add("group")
    for i, sel in enumerate(ast.expressions):
        for col in sel.find_all(exp.Column):
            if col.table == alias:
                sig.setdefault(col.name, set()).add(f"select_{i}")
    return sig


def compute_table_mapping(working_ast, reference_ast) -> Dict[str, str]:
    """Compute bijective alias mapping m : Aliases(Q★) → Aliases(Q).

    For tables without self-joins, the mapping is trivial by table name.
    For self-joins, we use column-usage signatures and greedy matching.
    Returns {ref_alias: working_alias}.
    """
    w_tables = _extract_tables_with_aliases(working_ast)
    r_tables = _extract_tables_with_aliases(reference_ast)

    w_by_table: Dict[str, List[str]] = {}
    for name, alias in w_tables:
        w_by_table.setdefault(name, []).append(alias)
    r_by_table: Dict[str, List[str]] = {}
    for name, alias in r_tables:
        r_by_table.setdefault(name, []).append(alias)

    mapping: Dict[str, str] = {}

    for table_name in set(r_by_table) & set(w_by_table):
        r_aliases = r_by_table[table_name]
        w_aliases = w_by_table[table_name]

        if len(r_aliases) == 1 and len(w_aliases) == 1:
            mapping[r_aliases[0]] = w_aliases[0]
            continue

        # Self-join: match by signature similarity
        r_sigs = {a: _build_alias_signature(reference_ast, a) for a in r_aliases}
        w_sigs = {a: _build_alias_signature(working_ast, a) for a in w_aliases}

        used_w = set()
        for ra in r_aliases:
            best_wa, best_score = None, -1
            for wa in w_aliases:
                if wa in used_w:
                    continue
                score = 0
                for col in set(r_sigs[ra]) & set(w_sigs[wa]):
                    score += len(r_sigs[ra][col] & w_sigs[wa][col])
                if score > best_score:
                    best_score = score
                    best_wa = wa
            if best_wa is not None:
                mapping[ra] = best_wa
                used_w.add(best_wa)

    return mapping


# ═══════════════════════════════════════════════════════════════════════════
# Predicate AST utilities
# ═══════════════════════════════════════════════════════════════════════════

def _extract_predicates(node):
    """Split AND-connected predicates into a flat list of AST nodes."""
    if isinstance(node, exp.And):
        return _extract_predicates(node.this) + _extract_predicates(node.expression)
    return [node]


def _predicate_columns(node) -> Set[str]:
    """Column names referenced in an AST predicate node."""
    return {c.name for c in node.find_all(exp.Column)}


def _ast_size(node) -> int:
    """Number of nodes in a sqlglot expression tree."""
    count = 1
    for child in node.iter_expressions():
        count += _ast_size(child)
    return count


def _extract_table_names(ast) -> Set[str]:
    """Set of table names from a SELECT AST."""
    tables = set()
    from_clause = ast.find(exp.From)
    if from_clause and from_clause.this:
        tables.add(from_clause.this.name)
    for j in ast.find_all(exp.Join):
        if j.this:
            tables.add(j.this.name)
    return tables


# ═══════════════════════════════════════════════════════════════════════════
# Repair cost (QR-Hint Definition 3)
# ═══════════════════════════════════════════════════════════════════════════

_W_SITE = 1 / 6   # penalty per repair site


def repair_cost(sites: list, fixes: list, p_size: int, p_star_size: int) -> float:
    """Cost(S, F) = w·|S| + Σ dist(s, F(s)) / (|P| + |P★|)."""
    denom = max(p_size + p_star_size, 1)
    dist_sum = sum(_ast_size(s) + _ast_size(f) for s, f in zip(sites, fixes))
    return _W_SITE * len(sites) + dist_sum / denom


# ═══════════════════════════════════════════════════════════════════════════
# Propagator-based equivalence and viability testing
#
# Instead of an external SMT solver, we use the propagator network's own
# cells and merge operations for equivalence / contradiction testing:
#
#   IsEquiv(e1, e2) → put both into the same cell; merge detects conflict
#   IsSatisfiable(P) → build a constraint network, check for contradiction
#   CreateBounds    → backward propagation through bidirectional constraints
# ═══════════════════════════════════════════════════════════════════════════

def _outputs_equivalent(out1, out2) -> bool:
    """Check if two FullRelation outputs are bag-equivalent.

    Actually uses the propagator merge: canonicalize both relations, then
    merge them.  If merge returns the_contradiction, they differ.
    """
    if out1 is None and out2 is None:
        return True
    if out1 is None or out2 is None:
        return False
    if not (isinstance(out1, FullRelation) and isinstance(out2, FullRelation)):
        return False
    # Canonicalize: strip alias-qualified keys, sort rows for deterministic comparison
    n1 = _canonicalize_relation(out1)
    n2 = _canonicalize_relation(out2)
    result = merge(n1, n2)
    return not contradictory_p(result)


def _canonicalize_relation(rel: FullRelation) -> FullRelation:
    """Strip alias-prefixed keys and sort rows for merge-based comparison."""
    canon_rows = sorted(
        (tuple(sorted((k, v) for k, v in row.items() if '.' not in k))
         for row in rel.rows),
        key=lambda t: t,
    )
    # Use tuples for frozen/hashable representation in FullRelation
    return FullRelation(rel.columns, [dict(r) for r in canon_rows])


def _test_predicate_equivalence(pred1_fn, pred2_fn, rows) -> bool:
    """Test if two predicate functions produce the same result on given rows.

    Uses the propagator merge to detect disagreements: for each row,
    evaluate both predicates and merge the boolean results.  A contradiction
    means the predicates disagree on that row.
    """
    for row in rows:
        try:
            r1 = bool(pred1_fn(row))
        except Exception:
            r1 = None
        try:
            r2 = bool(pred2_fn(row))
        except Exception:
            r2 = None
        if r1 is not None and r2 is not None:
            result = merge(r1, r2)
            if contradictory_p(result):
                return False
        elif r1 != r2:
            return False
    return True


def _run_query_network(sql, catalog):
    """Parse, build, and run a query network.  Returns (network, output)."""
    from ..scheduler import initialize_scheduler, run as scheduler_run
    from .parser import parse_query

    initialize_scheduler()
    net = parse_query(sql, catalog)
    scheduler_run()
    output = net.output_cell.content if net.output_cell else None
    return net, output


# ═══════════════════════════════════════════════════════════════════════════
# Backward propagation for repair-site identification
#
# QR-Hint §5.1 CreateBounds — implemented via the bidirectional constraint
# network.  When we inject the reference output into the output cell, the
# backward propagators push information toward the input cells.  Comparing
# this backward-propagated information with the working cell contents
# reveals where the divergence originates.
# ═══════════════════════════════════════════════════════════════════════════

def _identify_divergence_cells(working_net: QueryNetwork,
                               reference_net: QueryNetwork
                               ) -> List[Tuple[str, str, object, object]]:
    """Walk both networks and find cells where outputs diverge.

    Returns [(cell_name, op_type, working_content, reference_content), …]
    sorted from earliest (closest to scan) to latest (closest to output).
    """
    divergences = []
    w_cells = {name: cell.content for name, cell in working_net.cells.items()}
    r_cells = {name: cell.content for name, cell in reference_net.cells.items()}

    for op_type, op_info in working_net.operators:
        out_cell = op_info.get('output')
        if out_cell is None:
            continue
        cell_name = _find_cell_name(working_net, out_cell)
        if cell_name is None or cell_name not in r_cells:
            continue
        w_val = w_cells.get(cell_name)
        r_val = r_cells.get(cell_name)
        if w_val is None or r_val is None:
            continue
        if isinstance(w_val, FullRelation) and isinstance(r_val, FullRelation):
            if not _outputs_equivalent(w_val, r_val):
                divergences.append((cell_name, op_type, w_val, r_val))
    return divergences


# ═══════════════════════════════════════════════════════════════════════════
# TMS-based repair-site search (QR-Hint §5 RepairWhere)
#
# Uses the propagator TMS (Truth Maintenance System) to explore worldviews
# where different subsets of predicates are swapped between working and
# reference versions.  Each predicate position has two premises (w_i and
# r_i); TMS cells hold both supported values.  kick_out / bring_in switch
# worldviews; strongest_consequence reads the believed value per cell.
# ═══════════════════════════════════════════════════════════════════════════

def _safe_compile(p):
    """Compile a sqlglot predicate AST into a row → bool function."""
    from .parser import _compile_expr
    try:
        return _compile_expr(p)
    except Exception:
        return lambda r: True


def _safe_eval(fn, row):
    """Evaluate a predicate function on a row, defaulting to True on error."""
    try:
        return bool(fn(row))
    except Exception:
        return True


def _find_minimal_predicate_repairs(
    working_preds: List,
    reference_preds: List,
    all_rows: List[dict],
) -> Tuple[List[int], float]:
    """    
    Find minimal set of predicate indices whose replacement fixes WHERE.

    Uses the actual propagator TMS for worldview exploration:

    1. For each predicate position i, for each row j, build a TMS cell
       holding two supported boolean values:
         - Supported(working_result, {w_i})
         - Supported(reference_result, {r_i})

    2. Explore worldviews by calling kick_out / bring_in on premises.
       For a candidate repair set S, kick out {w_i : i ∈ S} and bring in
       {r_i : i ∈ S}.  The TMS caches invalidate on worldview change.

    3. Read each cell via strongest_consequence (which returns the believed
       supported value) and check if the combined WHERE output matches the
       reference — the propagator merge viability test.

    Returns (repair_site_indices, cost).
    """
    result = _find_minimal_predicate_repairs_tms(
        working_preds,
        reference_preds,
        all_rows,
    )
    return list(result.repair_sites), float(result.cost)


def _find_minimal_predicate_repairs_tms(
    working_preds: List,
    reference_preds: List,
    all_rows: List[dict],
) -> ApproachRepairResult:
    """TMS worldview search over predicate positions (existing approach)."""
    from ..scheduler import initialize_scheduler, run as scheduler_run
    from ..tms import (
        make_tms, kick_out, bring_in, strongest_consequence,
        mark_premise_out,
    )
    from ..supported_values import supported, supported_p

    n_w = len(working_preds)
    n_r = len(reference_preds)
    n_pos = max(n_w, n_r)

    if not all_rows or n_pos == 0:
        return ApproachRepairResult(
            approach="network_tms",
            repair_sites=[],
            cost=0.0,
            viable=True,
        )

    # ── Forward pass: evaluate predicates on all rows ──

    w_compiled = [_safe_compile(p) for p in working_preds]
    r_compiled = [_safe_compile(p) for p in reference_preds]

    w_vectors = [[_safe_eval(fn, row) for row in all_rows] for fn in w_compiled]
    r_vectors = [[_safe_eval(fn, row) for row in all_rows] for fn in r_compiled]

    # Reference combined result (AND of all reference predicates)
    ref_combined = [all(r_vectors[j][i] for j in range(n_r))
                    for i in range(len(all_rows))]

    # ── Identify mismatched predicate positions ──

    mismatched = []
    for i in range(min(n_w, n_r)):
        if w_vectors[i] != r_vectors[i]:
            mismatched.append(i)
    if n_w > n_r:
        mismatched.extend(range(n_r, n_w))
    elif n_r > n_w:
        mismatched.extend(range(n_w, n_r))

    if not mismatched:
        return ApproachRepairResult(
            approach="network_tms",
            repair_sites=[],
            cost=0.0,
            viable=True,
        )

    # ── Build TMS cells: one per (predicate position, row) ──
    #
    # Each cell holds two supported boolean values.  With all premises out
    # during setup, no contradictions arise (strongest_consequence returns
    # None when no premises are believed).

    initialize_scheduler()

    w_premises = [f"w_{i}" for i in range(n_w)]
    r_premises = [f"r_{i}" for i in range(n_r)]
    all_premises = w_premises + r_premises

    # Start with all premises out to avoid contradictions during cell setup
    for p in all_premises:
        mark_premise_out(p)

    cells = []   # cells[i][j] for predicate position i, row j
    for i in range(n_pos):
        row_cells = []
        for j in range(len(all_rows)):
            cell = Cell(name=f"p{i}_r{j}")
            if i < n_w:
                cell.add_content(
                    make_tms(supported(w_vectors[i][j], [w_premises[i]]))
                )
            if i < n_r:
                cell.add_content(
                    make_tms(supported(r_vectors[i][j], [r_premises[i]]))
                )
            row_cells.append(cell)
        cells.append(row_cells)

    scheduler_run()

    # ── TMS worldview exploration (QR-Hint Algorithm 1) ──
    #
    # For each candidate repair set, switch worldview via kick_out / bring_in,
    # then read cells with strongest_consequence to test viability.

    p_size = sum(_ast_size(p) for p in working_preds)
    p_star_size = sum(_ast_size(p) for p in reference_preds)

    best_sites = list(mismatched)
    best_cost = float('inf')
    found_viable = False

    for k in range(1, len(mismatched) + 1):
        if _W_SITE * k >= best_cost:
            break

        for subset in combinations(mismatched, k):
            # Skip if swapped position has no reference counterpart
            if any(idx >= n_r for idx in subset):
                continue

            # ── Worldview switch ──
            # Non-swapped positions: working premise in, reference out
            # Swapped positions:     working premise out, reference in
            subset_set = set(subset)
            for i in range(n_w):
                if i in subset_set:
                    kick_out(w_premises[i])
                else:
                    bring_in(w_premises[i])
            for i in range(n_r):
                if i in subset_set:
                    bring_in(r_premises[i])
                else:
                    kick_out(r_premises[i])
            # Note: we do NOT run the scheduler here — we only need to read
            # TMS cells via strongest_consequence, which uses the worldview
            # counter (invalidated by kick_out/bring_in) to re-evaluate.

            # ── Viability check via TMS cell queries ──
            equivalent = True
            for j in range(len(all_rows)):
                row_pass = True
                for i in range(n_pos):
                    cell = cells[i][j]
                    val = strongest_consequence(cell.content)
                    if supported_p(val):
                        val = val.value
                    if val is None:
                        # Position has no believed value (e.g., extra predicate
                        # with premise out) — treat as True (no filtering)
                        continue
                    if not val:
                        row_pass = False
                        break

                if row_pass != ref_combined[j]:
                    equivalent = False
                    break

            if equivalent:
                sites_ast = [working_preds[i] for i in subset if i < n_w]
                fixes_ast = [reference_preds[i] for i in subset if i < n_r]
                cost = repair_cost(sites_ast, fixes_ast, p_size, p_star_size) if sites_ast else 0.0
                if cost < best_cost:
                    best_cost = cost
                    best_sites = list(subset)
                    found_viable = True

        if best_cost < float('inf'):
            break  # Found minimum-cost fix at this cardinality

    if best_cost == float('inf'):
        best_cost = 0.0

    return ApproachRepairResult(
        approach="network_tms",
        repair_sites=best_sites,
        cost=best_cost,
        viable=found_viable or not mismatched,
        details={
            "candidate_positions": mismatched,
            "predicate_count_working": n_w,
            "predicate_count_reference": n_r,
        },
    )


def _row_signature(row: dict) -> Tuple[Tuple[str, Any], ...]:
    """Canonical row signature used for row-set equivalence checks."""
    return tuple(sorted((k, v) for k, v in row.items() if '.' not in k))


def _reference_mask_from_relation(all_rows: List[dict], rel: Optional[FullRelation]) -> List[bool]:
    """Build row-membership mask for relation rows over all_rows."""
    if rel is None:
        return [False] * len(all_rows)
    rel_set = {_row_signature(r) for r in rel.rows}
    return [_row_signature(r) in rel_set for r in all_rows]


def _evaluate_predicate_vector(preds: List, rows: List[dict]) -> List[List[bool]]:
    """Compile/evaluate each predicate into a boolean vector over rows."""
    compiled = [_safe_compile(p) for p in preds]
    return [[_safe_eval(fn, row) for row in rows] for fn in compiled]


def _combined_mask_from_vectors(vectors: List[List[bool]], positions: List[int], row_count: int) -> List[bool]:
    """AND-combine selected predicate vectors for each row."""
    if not positions:
        return [True] * row_count
    out = []
    for j in range(row_count):
        out.append(all(vectors[i][j] for i in positions))
    return out


def _find_minimal_predicate_repairs_from_network(
    working_preds: List,
    reference_preds: List,
    all_rows: List[dict],
    working_filtered: Optional[FullRelation],
    reference_filtered: Optional[FullRelation],
) -> ApproachRepairResult:
    """Find minimal predicate repairs using repair-network relation data.

    This approach avoids rebuilding networks and uses existing stage outputs:
    - `reference_filtered` gives the target row-membership mask.
    - Candidate subsets swap predicate vectors from working to reference.
    """
    n_w = len(working_preds)
    n_r = len(reference_preds)

    if not all_rows:
        return ApproachRepairResult(
            approach="network_data",
            repair_sites=[],
            cost=0.0,
            viable=True,
        )

    w_vectors = _evaluate_predicate_vector(working_preds, all_rows)
    r_vectors = _evaluate_predicate_vector(reference_preds, all_rows)

    if n_w == 0 and n_r == 0:
        return ApproachRepairResult(
            approach="network_data",
            repair_sites=[],
            cost=0.0,
            viable=True,
        )

    target_mask = _reference_mask_from_relation(all_rows, reference_filtered)

    mismatched = []
    for i in range(min(n_w, n_r)):
        if w_vectors[i] != r_vectors[i]:
            mismatched.append(i)
    if n_w > n_r:
        mismatched.extend(range(n_r, n_w))
    elif n_r > n_w:
        mismatched.extend(range(n_w, n_r))

    if not mismatched:
        return ApproachRepairResult(
            approach="network_data",
            repair_sites=[],
            cost=0.0,
            viable=True,
            details={"target_rows": sum(target_mask)},
        )

    p_size = sum(_ast_size(p) for p in working_preds)
    p_star_size = sum(_ast_size(p) for p in reference_preds)

    best_sites = list(mismatched)
    best_cost = float("inf")
    found_viable = False

    for k in range(1, len(mismatched) + 1):
        if _W_SITE * k >= best_cost:
            break
        for subset in combinations(mismatched, k):
            subset_set = set(subset)

            vectors = []
            for i in range(n_w):
                if i in subset_set and i < n_r:
                    vectors.append(r_vectors[i])
                else:
                    vectors.append(w_vectors[i])

            out_mask = _combined_mask_from_vectors(vectors, list(range(len(vectors))), len(all_rows))
            equivalent = out_mask == target_mask

            if equivalent:
                sites_ast = [working_preds[i] for i in subset if i < n_w]
                fixes_ast = [reference_preds[i] for i in subset if i < n_r]
                cost = repair_cost(sites_ast, fixes_ast, p_size, p_star_size) if sites_ast else 0.0
                if cost < best_cost:
                    best_cost = cost
                    best_sites = list(subset)
                    found_viable = True

        if found_viable:
            break

    if best_cost == float("inf"):
        best_cost = 0.0

    return ApproachRepairResult(
        approach="network_data",
        repair_sites=best_sites,
        cost=best_cost,
        viable=found_viable or not mismatched,
        details={
            "candidate_positions": mismatched,
            "target_rows": sum(target_mask),
            "working_rows": len(working_filtered.rows) if isinstance(working_filtered, FullRelation) else None,
            "reference_rows": len(reference_filtered.rows) if isinstance(reference_filtered, FullRelation) else None,
        },
    )


def _enumerate_subtree_sites(node, path: Tuple[int, ...] = ()) -> List[Tuple[Tuple[int, ...], Any]]:
    """Enumerate all subtree repair sites as (path, subtree)."""
    sites = [(path, node)]
    children = list(node.iter_expressions())
    for i, child in enumerate(children):
        sites.extend(_enumerate_subtree_sites(child, path + (i,)))
    return sites


def _node_at_path(root, path: Tuple[int, ...]):
    """Get subtree by child-index path."""
    node = root
    for idx in path:
        children = list(node.iter_expressions())
        if idx >= len(children):
            return None
        node = children[idx]
    return node


def _replace_subtree_at_path(root, path: Tuple[int, ...], new_subtree):
    """Return copy of root with subtree at path replaced by new_subtree."""
    if not path:
        return new_subtree.copy()

    new_root = root.copy()
    parent = new_root
    for idx in path[:-1]:
        children = list(parent.iter_expressions())
        if idx >= len(children):
            return new_root
        parent = children[idx]

    child_idx = path[-1]
    children = list(parent.iter_expressions())
    if child_idx >= len(children):
        return new_root
    target_child = children[child_idx]

    for key, value in parent.args.items():
        if value is target_child:
            parent.set(key, new_subtree.copy())
            return new_root
        if isinstance(value, list):
            for i, item in enumerate(value):
                if item is target_child:
                    new_list = list(value)
                    new_list[i] = new_subtree.copy()
                    parent.set(key, new_list)
                    return new_root
    return new_root


def _semantic_family(node) -> str:
    """Coarse predicate family used for QR-Hint bound compatibility."""
    if isinstance(node, (exp.And, exp.Or, exp.Not)):
        return "boolean"
    if isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.In)):
        return "comparison"
    if isinstance(node, exp.Column):
        return "column"
    if isinstance(node, exp.Literal):
        return "literal"
    return type(node).__name__


def _create_bounds_for_site(site_node, reference_root) -> List:
    """QR-Hint CreateBounds-style candidate bounds for one repair site."""
    site_cols = _predicate_columns(site_node)
    site_family = _semantic_family(site_node)
    bounds = []

    for _, ref_sub in _enumerate_subtree_sites(reference_root):
        ref_cols = _predicate_columns(ref_sub)
        if site_family == _semantic_family(ref_sub) and (
            not site_cols or not ref_cols or site_cols == ref_cols
        ):
            bounds.append(ref_sub)

    if reference_root not in bounds:
        bounds.append(reference_root)

    # Deduplicate by SQL string while preserving order.
    uniq = []
    seen = set()
    for b in bounds:
        k = b.sql()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(b)
    return uniq


def _eval_predicate_mask(pred, rows: List[dict]) -> List[bool]:
    """Evaluate one predicate AST as a mask over rows."""
    fn = _safe_compile(pred)
    return [_safe_eval(fn, row) for row in rows]


def _find_minimal_predicate_repairs_qr_hint(
    working_pred,
    reference_pred,
    all_rows: List[dict],
) -> ApproachRepairResult:
    """Paper-style RepairWhere/CreateBounds/DeriveFixes over AST sites.

    This path works directly on predicate subtrees:
    - Repair sites are AST subtree paths.
    - Bounds are generated from compatible reference subtrees.
    - Fixes are chosen by testing viability against the reference mask and
      minimizing Definition-3 cost.
    """
    if working_pred is None or reference_pred is None:
        return ApproachRepairResult(
            approach="qr_hint_ast",
            repair_sites=[],
            cost=0.0,
            viable=False,
        )
    if not all_rows:
        return ApproachRepairResult(
            approach="qr_hint_ast",
            repair_sites=[],
            cost=0.0,
            viable=True,
        )

    target_mask = _eval_predicate_mask(reference_pred, all_rows)
    working_mask = _eval_predicate_mask(working_pred, all_rows)
    if working_mask == target_mask:
        return ApproachRepairResult(
            approach="qr_hint_ast",
            repair_sites=[],
            cost=0.0,
            viable=True,
        )

    sites = _enumerate_subtree_sites(working_pred)
    # Exclude the root from first pass to favor localized fixes.
    local_sites = [(p, n) for p, n in sites if p]
    if not local_sites:
        local_sites = sites

    bounds_by_path = {
        path: _create_bounds_for_site(node, reference_pred)
        for path, node in local_sites
    }

    p_size = _ast_size(working_pred)
    p_star_size = _ast_size(reference_pred)

    best_cost = float("inf")
    best_paths: List[Tuple[int, ...]] = []
    best_fixes = []

    max_sites = min(3, len(local_sites))
    for k in range(1, max_sites + 1):
        if _W_SITE * k >= best_cost:
            break
        for subset in combinations(local_sites, k):
            paths = [p for p, _ in subset]
            candidate_lists = [bounds_by_path[p] for p in paths]
            if not all(candidate_lists):
                continue

            # DeriveFixes: search minimal viable bound assignment for sites.
            def _search(i: int, cur_pred, chosen):
                nonlocal best_cost, best_paths, best_fixes
                if i == len(paths):
                    out_mask = _eval_predicate_mask(cur_pred, all_rows)
                    if out_mask != target_mask:
                        return
                    sites_ast = [_node_at_path(working_pred, p) for p in paths]
                    cost = repair_cost(sites_ast, chosen, p_size, p_star_size)
                    if cost < best_cost:
                        best_cost = cost
                        best_paths = list(paths)
                        best_fixes = [c.copy() for c in chosen]
                    return

                path = paths[i]
                for bound in candidate_lists[i]:
                    next_pred = _replace_subtree_at_path(cur_pred, path, bound)
                    _search(i + 1, next_pred, chosen + [bound])

            _search(0, working_pred, [])

        if best_cost < float("inf"):
            break

    if best_cost == float("inf"):
        return ApproachRepairResult(
            approach="qr_hint_ast",
            repair_sites=[],
            cost=0.0,
            viable=False,
            details={"searched_sites": len(local_sites)},
        )

    return ApproachRepairResult(
        approach="qr_hint_ast",
        repair_sites=best_paths,
        cost=best_cost,
        viable=True,
        details={
            "fixes": [f.sql() for f in best_fixes],
            "searched_sites": len(local_sites),
        },
    )


def _compare_where_repair_approaches(
    working_preds: List,
    reference_preds: List,
    working_pred,
    reference_pred,
    all_rows: List[dict],
    working_filtered: Optional[FullRelation],
    reference_filtered: Optional[FullRelation],
) -> Dict[str, Any]:
    """Run side-by-side WHERE repair strategies and choose a default."""
    network_data = _find_minimal_predicate_repairs_from_network(
        working_preds,
        reference_preds,
        all_rows,
        working_filtered,
        reference_filtered,
    )
    network_tms = _find_minimal_predicate_repairs_tms(
        working_preds,
        reference_preds,
        all_rows,
    )
    qr_hint_ast = _find_minimal_predicate_repairs_qr_hint(
        working_pred,
        reference_pred,
        all_rows,
    )

    options = [network_data, network_tms, qr_hint_ast]
    viable = [o for o in options if o.viable]
    selected = min(viable, key=lambda o: (o.cost, len(o.repair_sites))) if viable else network_data

    return {
        "selected": selected,
        "network_data": network_data,
        "network_tms": network_tms,
        "qr_hint_ast": qr_hint_ast,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Repair Network — propagator network for stage-by-stage comparison
#
# Each SQL pipeline stage (FROM, WHERE, GROUP BY, SELECT, …) gets a triple
# of cells in the repair network:
#
#   working_cell ──┐
#                  ├── [merge-based equiv propagator] ──→ equiv_cell
#   reference_cell ┘
#
# The equivalence propagator calls merge() on both cell contents.
# If merge returns the_contradiction, the stage fails.  The merge system
# dispatches by type (FullRelation, EstimateInfo, SchemaInfo) — no manual
# isinstance checks are needed for equivalence testing.
# ═══════════════════════════════════════════════════════════════════════════

# Map query-network operator types to stage names
_OP_TO_STAGE = {
    'filter': 'WHERE',
    'join': 'JOIN',
    'aggregate': 'GROUP BY',
    'project': 'SELECT',
    'sort': 'ORDER BY',
    'limit': 'LIMIT',
}

# Canonical stage evaluation order (QR-Hint §3.1)
STAGE_ORDER = ['FROM', 'JOIN', 'WHERE', 'GROUP BY', 'SELECT', 'ORDER BY', 'LIMIT']

# Stages only meaningful when their prerequisite passes
_STAGE_REQUIRES = {
    'WHERE': 'FROM',
    'GROUP BY': 'FROM',
}

# Stages that appear in stage_results (others only generate hints)
_RESULT_STAGES = {'FROM', 'WHERE', 'GROUP BY', 'SELECT'}


@dataclass
class StageResult:
    """Comparison result for one repair stage in the propagator network."""
    stage: str
    working_cell: Cell
    reference_cell: Cell
    equiv_cell: Cell

    @property
    def passed(self) -> Optional[bool]:
        """True if the stage outputs match, False if not, None if unevaluated."""
        content = self.equiv_cell.content
        return None if nothing_p(content) else content

    @property
    def structural_mismatch(self) -> bool:
        """True if one query has this clause and the other doesn't."""
        w_has = not nothing_p(self.working_cell.content)
        r_has = not nothing_p(self.reference_cell.content)
        return w_has != r_has


class RepairNetwork:
    """Propagator network comparing two query networks stage by stage.

    Each stage has three cells:
      - working_cell:   intermediate result from the working query network
      - reference_cell: intermediate result from the reference query network
      - equiv_cell:     True/False after merge-based comparison

    The equivalence is computed by merge-based propagators.  The merge system
    dispatches by type (FullRelation, EstimateInfo, SchemaInfo) and produces
    the_contradiction when values differ — no isinstance checks needed.
    """

    def __init__(self):
        self.stages: Dict[str, StageResult] = {}
        self.stage_order: List[str] = []
        self.output_stage: Optional[StageResult] = None

    def stage_results(self) -> Dict[str, bool]:
        """Return {stage: passed} for stages in _RESULT_STAGES.

        Respects prerequisites: e.g. WHERE is excluded if FROM failed.
        """
        results = {}
        for name in self.stage_order:
            if name not in _RESULT_STAGES:
                continue
            sr = self.stages.get(name)
            if not sr or nothing_p(sr.equiv_cell.content):
                continue
            
            # Respect stage prerequisites (QR-Hint §3.1)
            req = _STAGE_REQUIRES.get(name)
            if req in self.stages:
                if self.stages[req].equiv_cell.content is not True:
                    continue
            results[name] = sr.passed
        return results


    def first_divergence(self) -> Optional[StageResult]:
        """Return the first stage where working ≠ reference."""
        for name in self.stage_order:
            sr = self.stages.get(name)
            if sr and sr.passed is False:
                return sr
        return None


def _canonicalize_for_merge(val):
    """Prepare a cell value for merge-based comparison.

    FullRelations need alias-key stripping and row sorting to ensure
    semantically equal relations compare as equal through merge.
    All other types pass through unchanged — the merge system handles
    type dispatch.
    """
    if isinstance(val, FullRelation):
        return _canonicalize_relation(val)
    return val


def _wire_stage_equiv(w_cell: Cell, r_cell: Cell, eq_cell: Cell):
    """Wire a merge-based equivalence propagator for one stage.

    When both inputs have content, canonicalize and merge them.
    If merge produces the_contradiction, the stage fails.
    """
    def check_equivalence():
        w = w_cell.content
        r = r_cell.content
        if nothing_p(w) or nothing_p(r):
            return
        result = merge(_canonicalize_for_merge(w), _canonicalize_for_merge(r))
        eq_cell.add_content(not contradictory_p(result))

    propagator([w_cell, r_cell], check_equivalence)


def build_repair_network(w_net: QueryNetwork, r_net: QueryNetwork,
                         w_ast, r_ast) -> RepairNetwork:
    """Build a propagator network that compares two query networks.

    For each pipeline stage, creates a cell pair (working + reference)
    seeded with the already-computed intermediate results, and wires a
    merge-based equivalence propagator.  The network runs during
    construction; all stage results are immediately available.

    Args:
        w_net: Working query network (already evaluated)
        r_net: Reference query network (already evaluated)
        w_ast: Working query AST (for FROM table extraction)
        r_ast: Reference query AST (for FROM table extraction)
    """
    from ..scheduler import initialize_scheduler, run as scheduler_run

    initialize_scheduler()
    repair = RepairNetwork()

    # ── FROM stage: compare table multisets ──
    from_w = Cell(name="w_FROM")
    from_r = Cell(name="r_FROM")
    from_eq = Cell(name="eq_FROM")
    from_w.add_content(dict(_table_multiset(w_ast)))
    from_r.add_content(dict(_table_multiset(r_ast)))
    _wire_stage_equiv(from_w, from_r, from_eq)
    repair.stages['FROM'] = StageResult('FROM', from_w, from_r, from_eq)
    repair.stage_order.append('FROM')

    # ── Operator stages: compare cell outputs ──
    w_by_type: Dict[str, list] = {}
    for op_type, op_info in w_net.operators:
        w_by_type.setdefault(op_type, []).append(op_info)

    r_by_type: Dict[str, list] = {}
    for op_type, op_info in r_net.operators:
        r_by_type.setdefault(op_type, []).append(op_info)

    for stage in STAGE_ORDER[1:]:  # Skip FROM (handled above)
        op_types = [t for t, s in _OP_TO_STAGE.items() if s == stage]

        w_outs = []
        r_outs = []
        for ot in op_types:
            for info in w_by_type.get(ot, []):
                c = info.get('output')
                if c:
                    w_outs.append(c)
            for info in r_by_type.get(ot, []):
                c = info.get('output')
                if c:
                    r_outs.append(c)

        wc = Cell(name=f"w_{stage}")
        rc = Cell(name=f"r_{stage}")
        eq = Cell(name=f"eq_{stage}")

        # For single-operator stages, compare the output directly.
        # For multi-operator stages (JOIN), compare the last output
        # (which represents the combined result).
        w_out = w_outs[-1] if w_outs else None
        r_out = r_outs[-1] if r_outs else None

        if stage == 'GROUP BY':
            # GROUP BY semantics: compare grouping columns, not aggregate
            # results.  The aggregate cell output includes computed values
            # which differ for different aggregate functions (COUNT vs SUM).
            w_grp = w_ast.find(exp.Group)
            r_grp = r_ast.find(exp.Group)
            w_keys = frozenset(
                c.name for c in w_grp.expressions if isinstance(c, exp.Column)
            ) if w_grp else None
            r_keys = frozenset(
                c.name for c in r_grp.expressions if isinstance(c, exp.Column)
            ) if r_grp else None
            if w_keys is not None:
                wc.add_content(w_keys)
            if r_keys is not None:
                rc.add_content(r_keys)
        else:
            if w_out and not nothing_p(w_out.content):
                wc.add_content(w_out.content)
            if r_out and not nothing_p(r_out.content):
                rc.add_content(r_out.content)

        # Determine presence for structural mismatch detection.
        # For GROUP BY, presence is based on AST; for others, on cell outputs.
        if stage == 'GROUP BY':
            w_present = not nothing_p(wc.content)
            r_present = not nothing_p(rc.content)
        else:
            w_present = w_out is not None
            r_present = r_out is not None

        if not w_present and not r_present:
            eq.add_content(True)   # Neither has this stage — match
        elif not w_present or not r_present:
            eq.add_content(False)  # Structural mismatch
        else:
            _wire_stage_equiv(wc, rc, eq)

        repair.stages[stage] = StageResult(stage, wc, rc, eq)
        repair.stage_order.append(stage)

    # ── Output stage: compare final outputs ──
    out_w = Cell(name="w_OUTPUT")
    out_r = Cell(name="r_OUTPUT")
    out_eq = Cell(name="eq_OUTPUT")
    if w_net.output_cell and not nothing_p(w_net.output_cell.content):
        out_w.add_content(w_net.output_cell.content)
    if r_net.output_cell and not nothing_p(r_net.output_cell.content):
        out_r.add_content(r_net.output_cell.content)
    _wire_stage_equiv(out_w, out_r, out_eq)
    repair.output_stage = StageResult('OUTPUT', out_w, out_r, out_eq)

    scheduler_run()
    return repair


# ═══════════════════════════════════════════════════════════════════════════
# Stage hint generators (QR-Hint §4–§8)
#
# Each generator is called ONLY when the repair network says the stage
# failed.  Pass/fail decisions come from the network's merge-based
# equivalence propagators — the generators focus purely on producing
# graduated, non-leaking hints.
# ═══════════════════════════════════════════════════════════════════════════

def _generate_from_hints(sr: StageResult, working_ast, reference_ast,
                         report: RepairReport, max_level: int):
    """FROM hint generator.  Called when the repair network's FROM stage fails.

    The repair network already compared table multisets via merge.
    Here we generate per-table hints from the Counter diffs.
    """
    w_tables = _table_multiset(working_ast)
    r_tables = _table_multiset(reference_ast)

    for table in set(w_tables) | set(r_tables):
        w_count = w_tables.get(table, 0)
        r_count = r_tables.get(table, 0)
        if w_count == r_count:
            continue

        if w_count < r_count:
            diff = r_count - w_count
            hint = RepairHint(
                clause="FROM", severity="error", level=1,
                message="Your FROM clause is missing one or more tables.",
                _error_kind="missing_table",
            )
            if max_level >= 2:
                hint.level = 2
                hint.message = (
                    f"You need {diff} more occurrence(s) of a table in your FROM clause."
                )
            if max_level >= 3:
                hint.level = 3
                hint.message = (
                    f"Consider whether you need another reference to table '{table}' "
                    f"(you have {w_count}, need {r_count})."
                )
            report.hints.append(hint)
        else:
            diff = w_count - r_count
            hint = RepairHint(
                clause="FROM", severity="warning", level=1,
                message="Your FROM clause has extra table(s).",
                _error_kind="extra_table",
            )
            if max_level >= 2:
                hint.level = 2
                hint.message = (
                    f"You have {diff} extra occurrence(s) of a table in your FROM clause."
                )
            report.hints.append(hint)


def _generate_where_hints(sr: StageResult, working_ast, reference_ast,
                          catalog, report: RepairReport, max_level: int,
                          w_output, r_output):
    """WHERE hint generator.  Called when the repair network's WHERE stage fails.

    The repair network compared filter cell outputs via merge.
    Here we determine whether it's a structural mismatch (missing/extra WHERE)
    or a content mismatch (wrong predicates), and run TMS repair search for
    the latter.
    """
    w_where = working_ast.find(exp.Where)
    r_where = reference_ast.find(exp.Where)

    # ── Structural mismatch: missing / extra WHERE ──
    # The repair network detected the failure; here we classify it.
    if w_where is None and r_where is not None:
        hint = RepairHint(
            clause="WHERE", severity="error", level=1,
            message="Your query is missing a WHERE clause.",
            _error_kind="missing_where",
        )
        if max_level >= 2:
            hint.level = 2
            hint.message = "Your query returns unfiltered rows. A WHERE clause is needed."
        if max_level >= 3:
            hint.level = 3
            hint.message = (
                "Add a WHERE clause. Consider which rows in your result should be excluded."
            )
        report.hints.append(hint)
        return

    if w_where is not None and r_where is None:
        hint = RepairHint(
            clause="WHERE", severity="error", level=1,
            message="Your query has a WHERE clause that shouldn't be there.",
            _error_kind="extra_where",
            _working_fragment=w_where.this.sql(),
        )
        if max_level >= 2:
            hint.level = 2
            hint.message = (
                "Your WHERE clause filters out rows that should appear. Remove it."
            )
        report.hints.append(hint)
        return

    # ── Content mismatch: both have WHERE but filter outputs differ ──
    # The repair network's merge on filter cell contents detected contradiction.
    # Run TMS repair search to identify which predicates need fixing.
    w_preds = _extract_predicates(w_where.this)
    r_preds = _extract_predicates(r_where.this)
    from_rows = _gather_from_rows(working_ast, catalog)

    # Side-by-side approaches:
    # 1) network_data: use repair-network relation outputs directly
    # 2) network_tms: premise worldviews at predicate positions
    # 3) qr_hint_ast: paper-style site/bounds/fix search on AST subtrees
    approaches = _compare_where_repair_approaches(
        w_preds,
        r_preds,
        w_where.this,
        r_where.this,
        from_rows,
        sr.working_cell.content if isinstance(sr.working_cell.content, FullRelation) else None,
        sr.reference_cell.content if isinstance(sr.reference_cell.content, FullRelation) else None,
    )
    selected = approaches["selected"]
    report.approach_comparison["WHERE"] = {
        "selected": selected.approach,
        "network_data": {
            "viable": approaches["network_data"].viable,
            "cost": approaches["network_data"].cost,
            "repair_sites": approaches["network_data"].repair_sites,
        },
        "network_tms": {
            "viable": approaches["network_tms"].viable,
            "cost": approaches["network_tms"].cost,
            "repair_sites": approaches["network_tms"].repair_sites,
        },
        "qr_hint_ast": {
            "viable": approaches["qr_hint_ast"].viable,
            "cost": approaches["qr_hint_ast"].cost,
            "repair_sites": approaches["qr_hint_ast"].repair_sites,
        },
    }

    repair_indices = list(selected.repair_sites)
    cost = selected.cost

    if not repair_indices:
        hint = RepairHint(
            clause="WHERE", severity="error", level=1,
            message="There is an error in your WHERE clause.",
            _error_kind="wrong_where",
            _working_fragment=w_where.this.sql(),
            cost=cost,
        )
        if max_level >= 2:
            hint.level = 2
            _add_where_characterization(hint, w_output, r_output)
        report.hints.append(hint)
        return

    # ── Generate graduated hints from repair sites ──
    n_sites = len(repair_indices)

    hint = RepairHint(
        clause="WHERE", severity="error", level=1,
        message="There is an error in your WHERE clause.",
        _error_kind="wrong_where",
        _working_fragment=w_where.this.sql(),
        cost=cost,
        repair_sites=repair_indices,
    )

    if max_level >= 2:
        hint.level = 2
        _add_where_characterization(hint, w_output, r_output)
        if n_sites == 1:
            if selected.approach == "qr_hint_ast":
                hint.message += " A single predicate subtree needs to be fixed."
            else:
                hint.message += " A single predicate needs to be fixed."
        else:
            if selected.approach == "qr_hint_ast":
                hint.message += f" {n_sites} predicate subtree site(s) need to be fixed."
            else:
                hint.message += f" {n_sites} predicate(s) need to be fixed."

    if max_level >= 3:
        hint.level = 3
        # QR-Hint §5: narrow down to columns / operator types
        # Safe: we only describe the user's own predicates
        wrong_cols = set()
        error_types = set()
        for idx in repair_indices:
            if not isinstance(idx, int):
                continue
            if idx < len(w_preds):
                p = w_preds[idx]
                wrong_cols |= _predicate_columns(p)
                if idx < len(r_preds):
                    r_p = r_preds[idx]
                    if type(p).__name__ != type(r_p).__name__:
                        error_types.add("comparison_operator")
                    else:
                        error_types.add("literal_value")

        parts = []
        if wrong_cols:
            parts.append(f"predicate on column(s) {', '.join(sorted(wrong_cols))}")
        if "comparison_operator" in error_types:
            parts.append("check your comparison operator (=, >, <, >=, <=, !=)")
        if "literal_value" in error_types:
            parts.append("check the constant/threshold value")
        if parts:
            hint.message = f"In your WHERE clause, {'; '.join(parts)}."
        hint.message += f" (Selected strategy: {selected.approach})"

    report.hints.append(hint)


def _add_where_characterization(hint: RepairHint, w_output, r_output):
    """Level 2 characterization: too restrictive / permissive."""
    if isinstance(w_output, FullRelation) and isinstance(r_output, FullRelation):
        w_count = len(w_output.rows)
        r_count = len(r_output.rows)
        if w_count < r_count:
            hint.message = (
                "Your WHERE clause is too restrictive — it filters out rows "
                "that should appear in the result."
            )
        elif w_count > r_count:
            hint.message = (
                "Your WHERE clause is too permissive — it lets through rows "
                "that should be excluded."
            )
        else:
            hint.message = (
                "Your WHERE clause returns the right number of rows but the wrong ones."
            )


def _gather_from_rows(working_ast, catalog) -> List[dict]:
    """Get the cross-product rows from FROM tables for predicate evaluation.

    Uses the catalog data — the same data the propagator network's scan
    cells operate on.  For joins, produces the cross product so predicates
    can be tested on the full input space.
    """
    tables = _extract_tables_with_aliases(working_ast)
    if not tables:
        return []

    table_data = []
    for table_name, alias in tables:
        full = catalog.get_table(table_name, LatticeLevel.FULL)
        if full is None or not isinstance(full, FullRelation):
            return []
        rows_with_alias = []
        for row in full.rows:
            new_row = dict(row)
            for k, v in row.items():
                new_row[f"{alias}.{k}"] = v
            rows_with_alias.append(new_row)
        table_data.append(rows_with_alias)

    if len(table_data) == 1:
        return table_data[0]

    from itertools import product as iterproduct
    result = []
    for combo in iterproduct(*table_data):
        merged = {}
        for row in combo:
            merged.update(row)
        result.append(merged)
    return result


def _generate_group_by_hints(sr: StageResult, working_ast, reference_ast,
                             report: RepairReport, max_level: int):
    """GROUP BY hint generator.  Called when the repair network's GROUP BY stage fails.

    The repair network compared aggregate cell outputs via merge.
    Here we classify the mismatch and compute Δ⁻/Δ⁺ for hints.
    """
    w_group = working_ast.find(exp.Group)
    r_group = reference_ast.find(exp.Group)

    # Structural mismatch
    if w_group is None and r_group is not None:
        hint = RepairHint(
            clause="GROUP BY", severity="error", level=1,
            message="Your query is missing a GROUP BY clause.",
            _error_kind="missing_group_by",
        )
        if max_level >= 2:
            hint.level = 2
            n = len([c for c in r_group.expressions if isinstance(c, exp.Column)])
            hint.message = f"Your query needs a GROUP BY clause (grouping on {n} column(s))."
        if max_level >= 3:
            hint.level = 3
            hint.message = (
                "Add a GROUP BY clause. Consider which columns in "
                "your SELECT are not aggregated."
            )
        report.hints.append(hint)
        return

    if w_group is not None and r_group is None:
        hint = RepairHint(
            clause="GROUP BY", severity="error", level=1,
            message="Your query has a GROUP BY clause that shouldn't be there.",
            _error_kind="extra_group_by",
            _working_fragment=w_group.sql(),
        )
        if max_level >= 2:
            hint.level = 2
            hint.message = "Remove your GROUP BY — the result is not grouped."
        report.hints.append(hint)
        return

    # Content mismatch: both have GROUP BY but outputs differ.
    # Compute Δ⁻ (wrong) and Δ⁺ (missing) from column sets.
    w_cols = {c.name for c in w_group.expressions if isinstance(c, exp.Column)}
    r_cols = {c.name for c in r_group.expressions if isinstance(c, exp.Column)}
    delta_minus = w_cols - r_cols
    delta_plus = r_cols - w_cols

    hint = RepairHint(
        clause="GROUP BY", severity="error", level=1,
        message="Your GROUP BY columns are incorrect.",
        _error_kind="wrong_group_by",
        _working_fragment=", ".join(e.sql() for e in w_group.expressions),
    )

    if max_level >= 2:
        hint.level = 2
        if delta_minus and delta_plus:
            hint.message = (
                f"Your GROUP BY has wrong columns — "
                f"{len(delta_minus)} to remove, {len(delta_plus)} to add."
            )
        elif delta_minus:
            hint.message = f"Your GROUP BY has {len(delta_minus)} extra column(s)."
        else:
            hint.message = f"Your GROUP BY needs {len(delta_plus)} more column(s)."

    if max_level >= 3:
        hint.level = 3
        if delta_minus:
            hint.message = (
                f"Column(s) {', '.join(sorted(delta_minus))} in your GROUP BY "
                f"may not belong there."
            )
        elif delta_plus:
            hint.message = (
                f"Your GROUP BY needs {len(delta_plus)} more column(s). "
                f"Check your SELECT list for non-aggregated columns."
            )

    report.hints.append(hint)


def _generate_select_hints(sr: StageResult, working_ast, reference_ast,
                           report: RepairReport, max_level: int):
    """SELECT hint generator.  Called when the repair network's SELECT stage fails.

    The repair network compared project cell outputs via merge.
    Here we analyze expression-level diffs for graduated hints.
    """
    w_exprs = [e.sql() for e in working_ast.expressions]
    r_exprs = [e.sql() for e in reference_ast.expressions]

    w_set = set(w_exprs)
    r_set = set(r_exprs)

    missing = r_set - w_set
    extra = w_set - r_set

    hint = RepairHint(
        clause="SELECT", severity="error" if missing else "warning", level=1,
        message="Your SELECT list is incorrect.",
        _error_kind="wrong_select",
        _working_fragment=", ".join(w_exprs),
    )

    if max_level >= 2:
        hint.level = 2
        if missing and extra:
            hint.message = (
                f"Your SELECT list has {len(missing)} missing "
                f"and {len(extra)} extra expression(s)."
            )
        elif missing:
            hint.message = f"Your SELECT list is missing {len(missing)} expression(s)."
        else:
            hint.message = f"Your SELECT list has {len(extra)} extra expression(s)."

    if max_level >= 3:
        hint.level = 3
        if extra:
            hint.message = (
                f"Expression(s) in your SELECT that may not belong: "
                f"{', '.join(sorted(extra))}."
            )
        elif missing:
            has_agg = any(
                any(kw in m.upper() for kw in ('SUM(', 'COUNT(', 'AVG(', 'MIN(', 'MAX('))
                for m in missing
            )
            if has_agg:
                hint.message = (
                    f"Your SELECT is missing {len(missing)} aggregate expression(s). "
                    f"Check if you need SUM, COUNT, AVG, etc."
                )
            else:
                hint.message = f"Your SELECT is missing {len(missing)} column(s)."

    report.hints.append(hint)


def _generate_join_hints(sr: StageResult, working_ast, reference_ast,
                         report: RepairReport, max_level: int):
    """JOIN hint generator.  Called when the repair network's JOIN stage fails.

    The repair network compared join cell outputs via merge.
    Here we do per-join AST-level analysis for detailed hints.
    """
    w_joins = list(working_ast.find_all(exp.Join))
    r_joins = list(reference_ast.find_all(exp.Join))

    if len(w_joins) == 0 and len(r_joins) == 0:
        return

    if len(w_joins) != len(r_joins):
        hint = RepairHint(
            clause="JOIN", severity="error", level=1,
            message="Your query has the wrong number of JOINs.",
            _error_kind="wrong_join_count",
        )
        if max_level >= 2:
            hint.level = 2
            diff = len(w_joins) - len(r_joins)
            if diff > 0:
                hint.message = f"Your query has {diff} extra JOIN(s)."
            else:
                hint.message = f"Your query is missing {-diff} JOIN(s)."
        report.hints.append(hint)
        return

    for i, (wj, rj) in enumerate(zip(w_joins, r_joins)):
        w_on = wj.args.get("on")
        r_on = rj.args.get("on")
        w_on_sql = w_on.sql() if w_on else ""
        r_on_sql = r_on.sql() if r_on else ""

        w_table = wj.this.name if wj.this else "?"
        r_table = rj.this.name if rj.this else "?"

        if w_table != r_table:
            hint = RepairHint(
                clause="JOIN", severity="error", level=1,
                message=f"JOIN #{i+1} references the wrong table.",
                _error_kind="wrong_join_table",
                _working_fragment=w_table,
            )
            if max_level >= 2:
                hint.level = 2
                hint.message = (
                    f"JOIN #{i+1}: you are joining table '{w_table}', "
                    f"but a different table is expected."
                )
            report.hints.append(hint)

        elif w_on_sql != r_on_sql:
            w_cols = {c.name for c in (w_on.find_all(exp.Column) if w_on else [])}
            r_cols = {c.name for c in (r_on.find_all(exp.Column) if r_on else [])}

            hint = RepairHint(
                clause="JOIN", severity="error", level=1,
                message=f"JOIN #{i+1} ON condition is incorrect.",
                _error_kind="wrong_join_condition",
                _working_fragment=w_on_sql,
            )
            if max_level >= 2:
                hint.level = 2
                if w_cols != r_cols:
                    hint.message = (
                        f"JOIN #{i+1}: your ON condition references the wrong columns."
                    )
                else:
                    hint.message = (
                        f"JOIN #{i+1}: your ON condition uses the right columns "
                        f"but the wrong expression."
                    )
            if max_level >= 3 and w_cols != r_cols:
                hint.level = 3
                extra_cols = w_cols - r_cols
                if extra_cols:
                    hint.message = (
                        f"JOIN #{i+1}: column(s) {', '.join(extra_cols)} "
                        f"in your ON condition may not belong there."
                    )
            report.hints.append(hint)


def _generate_order_by_hints(sr: StageResult, working_ast, reference_ast,
                             report: RepairReport, max_level: int):
    """ORDER BY hint generator.  Called when the repair network's ORDER BY stage fails."""
    w_order = working_ast.find(exp.Order)
    r_order = reference_ast.find(exp.Order)

    if w_order is None and r_order is None:
        return
    if w_order is None and r_order is not None:
        report.hints.append(RepairHint(
            clause="ORDER BY", severity="warning", level=1,
            message="Your query is missing an ORDER BY clause.",
            _error_kind="missing_order_by",
        ))
    elif w_order is not None and r_order is None:
        report.hints.append(RepairHint(
            clause="ORDER BY", severity="warning", level=1,
            message="Your query has an ORDER BY clause that may not be needed.",
            _error_kind="extra_order_by",
            _working_fragment=w_order.sql(),
        ))
    elif w_order.sql() != r_order.sql():
        hint = RepairHint(
            clause="ORDER BY", severity="warning", level=1,
            message="Your ORDER BY clause is incorrect.",
            _error_kind="wrong_order_by",
            _working_fragment=w_order.sql(),
        )
        if max_level >= 2:
            hint.level = 2
            hint.message = (
                "Check the column(s) and direction (ASC/DESC) in your ORDER BY."
            )
        report.hints.append(hint)


def _generate_limit_hints(sr: StageResult, working_ast, reference_ast,
                          report: RepairReport, max_level: int):
    """LIMIT hint generator.  Called when the repair network's LIMIT stage fails."""
    w_limit = working_ast.find(exp.Limit)
    r_limit = reference_ast.find(exp.Limit)

    if w_limit is None and r_limit is None:
        return
    if w_limit is None and r_limit is not None:
        report.hints.append(RepairHint(
            clause="LIMIT", severity="warning", level=1,
            message="Your query is missing a LIMIT clause.",
            _error_kind="missing_limit",
        ))
    elif w_limit is not None and r_limit is None:
        report.hints.append(RepairHint(
            clause="LIMIT", severity="warning", level=1,
            message="Your query has a LIMIT clause that may not be needed.",
            _error_kind="extra_limit",
            _working_fragment=w_limit.sql(),
        ))
    elif w_limit.expression.sql() != r_limit.expression.sql():
        hint = RepairHint(
            clause="LIMIT", severity="error", level=1,
            message="Your LIMIT value is incorrect.",
            _error_kind="wrong_limit_value",
            _working_fragment=w_limit.sql(),
        )
        if max_level >= 2:
            hint.level = 2
            w_val = w_limit.expression.sql()
            hint.message = (
                f"Your LIMIT of {w_val} does not match the expected number of rows."
            )
        report.hints.append(hint)


# Hint generator dispatch table
# Bind hint generators (defined above) to stage names
def _make_stage_gen(fn, needs_catalog=False, needs_output=False):
    """Wrap a hint generator to match the uniform dispatch signature."""
    def wrapper(sr, w_ast, r_ast, catalog, report, max_level, w_output, r_output):
        if needs_catalog and needs_output:
            fn(sr, w_ast, r_ast, catalog, report, max_level, w_output, r_output)
        elif needs_catalog:
            fn(sr, w_ast, r_ast, catalog, report, max_level)
        else:
            fn(sr, w_ast, r_ast, report, max_level)
    return wrapper

_HINT_GENERATORS = {
    'FROM':     _make_stage_gen(_generate_from_hints),
    'JOIN':     _make_stage_gen(_generate_join_hints),
    'WHERE':    _make_stage_gen(_generate_where_hints, needs_catalog=True, needs_output=True),
    'GROUP BY': _make_stage_gen(_generate_group_by_hints),
    'SELECT':   _make_stage_gen(_generate_select_hints),
    'ORDER BY': _make_stage_gen(_generate_order_by_hints),
    'LIMIT':    _make_stage_gen(_generate_limit_hints),
}

# ═══════════════════════════════════════════════════════════════════════════
# Data-level divergence (reads from repair network)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_divergence_hints(repair_net: RepairNetwork,
                               report: RepairReport,
                               max_level: int):
    """Generate data-level divergence hint from the repair network.

    Finds the first stage where working ≠ reference (via the network's
    merge-based equivalence cells) and generates a row-count hint.
    This replaces the old cell-by-cell walk — the repair network already
    tracks per-stage divergence via propagators.
    """
    first = repair_net.first_divergence()
    if not first or first.stage == 'FROM':
        return  # FROM divergence is handled by _generate_from_hints

    w_val = first.working_cell.content
    r_val = first.reference_cell.content

    if not (isinstance(w_val, FullRelation) and isinstance(r_val, FullRelation)):
        return

    clause = first.stage

    w_set = {tuple(sorted((k, v) for k, v in r.items() if '.' not in k))
             for r in w_val.rows}
    r_set = {tuple(sorted((k, v) for k, v in r.items() if '.' not in k))
             for r in r_val.rows}
    missing = len(r_set - w_set)
    extra = len(w_set - r_set)

    hint = RepairHint(
        clause=clause, severity="error", level=1,
        message=f"Results first diverge at the {clause} step.",
        _error_kind="data_divergence",
        row_impact=missing + extra,
    )
    if max_level >= 2:
        hint.level = 2
        if missing > 0 and extra == 0:
            hint.message = (
                f"Your {clause} step produces too few rows "
                f"({len(w_set)} instead of {len(r_set)})."
            )
        elif extra > 0 and missing == 0:
            hint.message = (
                f"Your {clause} step produces too many rows "
                f"({len(w_set)} instead of {len(r_set)})."
            )
        else:
            hint.message = (
                f"Your {clause} step produces different rows "
                f"({missing} missing, {extra} extra)."
            )
    report.hints.append(hint)


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point — QR-Hint stage-based diagnosis
# ═══════════════════════════════════════════════════════════════════════════


def diagnose_from_reference(working_sql: str,
                            reference_sql: str,
                            catalog,
                            hint_level: HintLevel = HintLevel.DIRECTION,
                            ) -> RepairReport:
    """QR-Hint oracle: graduated, non-leaking repair hints.

    Builds a **repair network** — a propagator network that compares the
    working and reference query networks at each pipeline stage (FROM,
    WHERE, GROUP BY, SELECT, …).  Each stage's equivalence is determined
    by a merge-based propagator; no manual isinstance checks are needed.

    Stage results are read directly from the repair network's cells.
    Hint generators are called only for stages where the network detects
    a failure.

    The reference query is NEVER exposed in any hint.

    Args:
        working_sql:   Student's (potentially wrong) SQL query
        reference_sql: Correct reference SQL query (kept secret)
        catalog:       Catalog with table data
        hint_level:    Maximum hint detail (CLAUSE / CHARACTER / DIRECTION)

    Returns:
        RepairReport with graduated, non-leaking hints
    """
    report = RepairReport(
        working_sql=working_sql.strip(),
        _reference_sql=reference_sql.strip(),
    )

    # Parse ASTs for structural comparison (used by hint generators)
    working_ast = sqlglot.parse_one(working_sql)
    reference_ast = sqlglot.parse_one(reference_sql)

    # Build and run both propagator query networks
    w_net, w_output = _run_query_network(working_sql, catalog)
    r_net, r_output = _run_query_network(reference_sql, catalog)

    report._working_output = w_output
    report._reference_output = r_output

    # Safe row counts (never leaks reference data)
    if isinstance(w_output, FullRelation):
        report.working_row_count = len(w_output.rows)
    if isinstance(r_output, FullRelation):
        report.reference_row_count = len(r_output.rows)

    # ── Build repair network (propagator-based stage-by-stage comparison) ──
    repair_net = build_repair_network(w_net, r_net, working_ast, reference_ast)

    # Overall equivalence from the repair network's output stage
    if repair_net.output_stage and repair_net.output_stage.passed:
        report.equivalent = True
        report.hints.append(RepairHint(
            clause="RESULT", severity="info", level=1,
            message="Your query produces the correct result.",
        ))
        return report

    max_level = int(hint_level)

    # Read stage results from the repair network's equivalence cells
    report.stage_results = repair_net.stage_results()

    # Compute table mapping (needed by downstream stages)
    report._table_mapping = compute_table_mapping(working_ast, reference_ast)

    # ════ Generate hints for failing stages ════
    # Each stage is checked via the repair network's cells.
    # Hint generators are called only when the network says a stage failed.

    for stage_name in repair_net.stage_order:
        stage_result = repair_net.stages.get(stage_name)
        if not stage_result:
            continue
        if repair_net.stages[stage_name].passed is False:
            gen = _HINT_GENERATORS.get(stage_name)
            if gen:
                gen(stage_result, working_ast, reference_ast, catalog,
                    report, max_level, w_output, r_output)

    # Data-level divergence hint from the repair network
    _generate_divergence_hints(repair_net, report, max_level)
    return report




# ═══════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════

def print_repair_report(report: RepairReport, show_sql: bool = False):
    """Pretty-print a repair report. Never reveals reference query."""
    print(f"\n{'='*70}")
    print(f"  QR-Hint Repair Report")
    print(f"{'='*70}")

    if show_sql:
        print(f"\n  Your query: {report.working_sql}")

    if report.equivalent:
        print(f"\n  ✓ Your query produces the correct result!")
        return

    if report.working_row_count is not None and report.reference_row_count is not None:
        print(f"\n  Your output:    {report.working_row_count} rows")
        print(f"  Expected:       {report.reference_row_count} rows")

    if report.stage_results:
        print(f"\n  Stage Results:")
        for stage, passed in report.stage_results.items():
            icon = "✓" if passed else "✗"
            print(f"    {icon} {stage}")

    if report.approach_comparison:
        print(f"\n  Approach Comparison:")
        where_cmp = report.approach_comparison.get("WHERE")
        if where_cmp:
            print(f"    WHERE selected: {where_cmp.get('selected')}")
            for key in ("network_data", "network_tms", "qr_hint_ast"):
                item = where_cmp.get(key, {})
                print(
                    f"      - {key}: viable={item.get('viable')}, "
                    f"cost={item.get('cost')}, sites={item.get('repair_sites')}"
                )

    if not report.hints:
        print(f"\n  No hints available.")
        return

    errors = [h for h in report.hints if h.severity == "error"]
    warnings = [h for h in report.hints if h.severity == "warning"]
    infos = [h for h in report.hints if h.severity == "info"]

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for h in errors:
            cost_str = f" [cost={h.cost:.2f}]" if h.cost is not None else ""
            print(f"    [!] {h.clause}: {h.message}{cost_str}")

    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for h in warnings:
            print(f"    [?] {h.clause}: {h.message}")

    if infos:
        print(f"\n  Info:")
        for h in infos:
            print(f"    [i] {h.clause}: {h.message}")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _find_cell_name(network, cell):
    for name, c in network.cells.items():
        if c is cell:
            return name
    return None
