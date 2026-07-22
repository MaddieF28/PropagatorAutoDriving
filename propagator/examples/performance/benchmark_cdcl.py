#!/usr/bin/env python3
"""
CDCL vs DDB (Dependency-Directed Backtracking) Performance Comparison

This benchmark compares the performance of the propagator solver with:
1. DDB (original): Standard dependency-directed backtracking with pairwise_union
2. CDCL: Conflict-Driven Clause Learning with 1-UIP, backjumping, and VSIDS

Run with:
    python3 -m propagator.examples.benchmark_cdcl
"""

import time
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
import sys


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    method: str  # 'ddb' or 'cdcl'
    time_seconds: float
    contradictions: int
    solution_found: bool
    solution: Optional[dict] = None
    cdcl_decisions: int = 0
    cdcl_backjumps: int = 0
    cdcl_levels_saved: int = 0
    
    def summary(self) -> str:
        base = f"  {self.method:6s}: {self.time_seconds:.4f}s, {self.contradictions} conflicts"
        if self.method == 'cdcl':
            base += f", {self.cdcl_backjumps} backjumps"
            if self.cdcl_backjumps > 0:
                avg = self.cdcl_levels_saved / max(1, self.cdcl_backjumps)
                base += f", avg {avg:.1f} levels saved"
        return base


def run_multiple_dwelling_ddb() -> BenchmarkResult:
    """Run Multiple Dwelling puzzle with DDB (CDCL disabled)."""
    from propagator import (
        Cell, initialize_scheduler, disable_cdcl, 
        get_number_of_calls_to_fail, get_contradictions,
    )
    from propagator.primitives import eq, constant, gt, subtractor, absolute_value
    from propagator.guessing_machine import one_of, require, abhor, require_distinct
    from propagator.tms import tms_query, tms_p
    from propagator.scheduler import run
    
    initialize_scheduler()
    disable_cdcl()
    
    start = time.time()
    
    # Create cells
    baker = Cell(name='baker')
    cooper = Cell(name='cooper')
    fletcher = Cell(name='fletcher')
    miller = Cell(name='miller')
    smith = Cell(name='smith')
    
    floors = [1, 2, 3, 4, 5]
    
    # Each person lives on one of the floors
    one_of(floors, baker)
    one_of(floors, fletcher)
    one_of(floors, smith)
    one_of(floors, cooper)
    one_of(floors, miller)
    
    # Everyone lives on different floors
    require_distinct([baker, fletcher, smith, cooper, miller])
    
    # Constraint cells
    b_eq_5 = Cell('b_eq_5')
    f_eq_5 = Cell('f_eq_5')
    m_gt_c = Cell('m_gt_c')
    fc = Cell('fc')
    five = Cell('five')
    as_f = Cell('as_f')
    af_c = Cell('af_c')
    c_eq_1 = Cell('c_eq_1')
    f_eq_1 = Cell('f_eq_1')
    sf = Cell('sf')
    one = Cell('one')
    s_f = Cell('s_f')
    f_c = Cell('f_c')
    constant(1, one)
    constant(5, five)
    
    # Constraints
    eq(five, baker, b_eq_5); abhor(b_eq_5)
    eq(one, cooper, c_eq_1); abhor(c_eq_1)
    eq(five, fletcher, f_eq_5); abhor(f_eq_5)
    eq(one, fletcher, f_eq_1); abhor(f_eq_1)
    gt(miller, cooper, m_gt_c); require(m_gt_c)
    subtractor(smith, fletcher, s_f)
    absolute_value(s_f, as_f)
    eq(one, as_f, sf); abhor(sf)
    subtractor(fletcher, cooper, f_c)
    absolute_value(f_c, af_c)
    eq(one, af_c, fc); abhor(fc)
    
    run()
    
    elapsed = time.time() - start
    contradictions = len(get_contradictions())
    
    # Extract solution
    solution = {}
    for name, cell in [('baker', baker), ('cooper', cooper), ('fletcher', fletcher),
                       ('miller', miller), ('smith', smith)]:
        content = cell.content
        if tms_p(content):
            result = tms_query(content)
            if result is not None:
                val = result.value if hasattr(result, 'value') else result
                solution[name] = val
    
    return BenchmarkResult(
        name='Multiple Dwelling',
        method='ddb',
        time_seconds=elapsed,
        contradictions=contradictions,
        solution_found=len(solution) == 5,
        solution=solution,
    )


def run_multiple_dwelling_cdcl() -> BenchmarkResult:
    """Run Multiple Dwelling puzzle with CDCL enabled."""
    from propagator import (
        Cell, initialize_scheduler, enable_cdcl, 
        get_number_of_calls_to_fail, get_contradictions,
        cdcl_stats, get_cdcl_engine,
    )
    from propagator.primitives import eq, constant, gt, subtractor, absolute_value
    from propagator.guessing_machine import one_of, require, abhor, require_distinct
    from propagator.tms import tms_query, tms_p
    from propagator.scheduler import run
    
    initialize_scheduler()
    engine = enable_cdcl()
    
    start = time.time()
    
    # Create cells
    baker = Cell(name='baker')
    cooper = Cell(name='cooper')
    fletcher = Cell(name='fletcher')
    miller = Cell(name='miller')
    smith = Cell(name='smith')
    
    floors = [1, 2, 3, 4, 5]
    
    # Each person lives on one of the floors
    one_of(floors, baker)
    one_of(floors, fletcher)
    one_of(floors, smith)
    one_of(floors, cooper)
    one_of(floors, miller)
    
    # Everyone lives on different floors
    require_distinct([baker, fletcher, smith, cooper, miller])
    
    # Constraint cells
    b_eq_5 = Cell('b_eq_5')
    f_eq_5 = Cell('f_eq_5')
    m_gt_c = Cell('m_gt_c')
    fc = Cell('fc')
    five = Cell('five')
    as_f = Cell('as_f')
    af_c = Cell('af_c')
    c_eq_1 = Cell('c_eq_1')
    f_eq_1 = Cell('f_eq_1')
    sf = Cell('sf')
    one = Cell('one')
    s_f = Cell('s_f')
    f_c = Cell('f_c')
    constant(1, one)
    constant(5, five)
    
    # Constraints
    eq(five, baker, b_eq_5); abhor(b_eq_5)
    eq(one, cooper, c_eq_1); abhor(c_eq_1)
    eq(five, fletcher, f_eq_5); abhor(f_eq_5)
    eq(one, fletcher, f_eq_1); abhor(f_eq_1)
    gt(miller, cooper, m_gt_c); require(m_gt_c)
    subtractor(smith, fletcher, s_f)
    absolute_value(s_f, as_f)
    eq(one, as_f, sf); abhor(sf)
    subtractor(fletcher, cooper, f_c)
    absolute_value(f_c, af_c)
    eq(one, af_c, fc); abhor(fc)
    
    run()
    
    elapsed = time.time() - start
    contradictions = len(get_contradictions())
    
    # Extract solution
    solution = {}
    for name, cell in [('baker', baker), ('cooper', cooper), ('fletcher', fletcher),
                       ('miller', miller), ('smith', smith)]:
        content = cell.content
        if tms_p(content):
            result = tms_query(content)
            if result is not None:
                val = result.value if hasattr(result, 'value') else result
                solution[name] = val
    
    return BenchmarkResult(
        name='Multiple Dwelling',
        method='cdcl',
        time_seconds=elapsed,
        contradictions=contradictions,
        solution_found=len(solution) == 5,
        solution=solution,
        cdcl_decisions=engine.stats.decisions,
        cdcl_backjumps=engine.stats.backjumps,
        cdcl_levels_saved=engine.stats.backjump_levels_saved,
    )


def run_n_queens_ddb(n: int = 8) -> BenchmarkResult:
    """Run N-Queens puzzle with DDB."""
    from propagator import Cell, initialize_scheduler, disable_cdcl, get_contradictions
    from propagator.primitives import constant, eq, subtractor, absolute_value
    from propagator.guessing_machine import one_of, abhor, require_distinct
    from propagator.tms import tms_query, tms_p
    from propagator.scheduler import run
    
    initialize_scheduler()
    disable_cdcl()
    
    start = time.time()
    
    # Queens[i] = column of queen in row i
    queens = [Cell(name=f'queen_{i}') for i in range(n)]
    columns = list(range(n))
    
    # Each queen in some column
    for q in queens:
        one_of(columns, q)
    
    # All queens on different columns
    require_distinct(queens)
    
    # Diagonal constraints: |row_i - row_j| != |col_i - col_j|
    for i in range(n):
        for j in range(i + 1, n):
            row_diff = j - i  # Always positive
            col_diff = Cell(name=f'col_diff_{i}_{j}')
            abs_col_diff = Cell(name=f'abs_diff_{i}_{j}')
            row_const = Cell(name=f'row_const_{i}_{j}')
            diag_eq = Cell(name=f'diag_{i}_{j}')
            
            subtractor(queens[i], queens[j], col_diff)
            absolute_value(col_diff, abs_col_diff)
            constant(row_diff, row_const)
            eq(row_const, abs_col_diff, diag_eq)
            abhor(diag_eq)  # Diagonal attack not allowed
    
    run()
    
    elapsed = time.time() - start
    contradictions = len(get_contradictions())
    
    # Extract solution
    solution = {}
    for i, q in enumerate(queens):
        content = q.content
        if tms_p(content):
            result = tms_query(content)
            if result is not None:
                val = result.value if hasattr(result, 'value') else result
                solution[f'queen_{i}'] = val
    
    return BenchmarkResult(
        name=f'{n}-Queens',
        method='ddb',
        time_seconds=elapsed,
        contradictions=contradictions,
        solution_found=len(solution) == n,
        solution=solution,
    )


def run_n_queens_cdcl(n: int = 8) -> BenchmarkResult:
    """Run N-Queens puzzle with CDCL."""
    from propagator import Cell, initialize_scheduler, enable_cdcl, get_contradictions, get_cdcl_engine
    from propagator.primitives import constant, eq, subtractor, absolute_value
    from propagator.guessing_machine import one_of, abhor, require_distinct
    from propagator.tms import tms_query, tms_p
    from propagator.scheduler import run
    
    initialize_scheduler()
    engine = enable_cdcl()
    
    start = time.time()
    
    # Queens[i] = column of queen in row i
    queens = [Cell(name=f'queen_{i}') for i in range(n)]
    columns = list(range(n))
    
    # Each queen in some column
    for q in queens:
        one_of(columns, q)
    
    # All queens on different columns
    require_distinct(queens)
    
    # Diagonal constraints: |row_i - row_j| != |col_i - col_j|
    for i in range(n):
        for j in range(i + 1, n):
            row_diff = j - i
            col_diff = Cell(name=f'col_diff_{i}_{j}')
            abs_col_diff = Cell(name=f'abs_diff_{i}_{j}')
            row_const = Cell(name=f'row_const_{i}_{j}')
            diag_eq = Cell(name=f'diag_{i}_{j}')
            
            subtractor(queens[i], queens[j], col_diff)
            absolute_value(col_diff, abs_col_diff)
            constant(row_diff, row_const)
            eq(row_const, abs_col_diff, diag_eq)
            abhor(diag_eq)
    
    run()
    
    elapsed = time.time() - start
    contradictions = len(get_contradictions())
    
    # Extract solution
    solution = {}
    for i, q in enumerate(queens):
        content = q.content
        if tms_p(content):
            result = tms_query(content)
            if result is not None:
                val = result.value if hasattr(result, 'value') else result
                solution[f'queen_{i}'] = val
    
    return BenchmarkResult(
        name=f'{n}-Queens',
        method='cdcl',
        time_seconds=elapsed,
        contradictions=contradictions,
        solution_found=len(solution) == n,
        solution=solution,
        cdcl_decisions=engine.stats.decisions,
        cdcl_backjumps=engine.stats.backjumps,
        cdcl_levels_saved=engine.stats.backjump_levels_saved,
    )


def run_graph_coloring_ddb(num_nodes: int = 10, edge_prob: float = 0.3, num_colors: int = 3, seed: int = 42) -> BenchmarkResult:
    """Run graph coloring with DDB."""
    import random
    from propagator import Cell, initialize_scheduler, disable_cdcl, get_contradictions
    from propagator.primitives import eq
    from propagator.guessing_machine import one_of, abhor
    from propagator.tms import tms_query, tms_p
    from propagator.scheduler import run
    
    random.seed(seed)
    
    # Generate random graph
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                edges.append((i, j))
    
    initialize_scheduler()
    disable_cdcl()
    
    start = time.time()
    
    # Node color cells
    nodes = [Cell(name=f'node_{i}') for i in range(num_nodes)]
    colors = list(range(num_colors))
    
    # Each node has a color
    for node in nodes:
        one_of(colors, node)
    
    # Adjacent nodes have different colors
    for i, j in edges:
        eq_cell = Cell(name=f'eq_{i}_{j}')
        eq(nodes[i], nodes[j], eq_cell)
        abhor(eq_cell)
    
    run()
    
    elapsed = time.time() - start
    contradictions = len(get_contradictions())
    
    # Extract solution
    solution = {}
    for i, node in enumerate(nodes):
        content = node.content
        if tms_p(content):
            result = tms_query(content)
            if result is not None:
                val = result.value if hasattr(result, 'value') else result
                solution[f'node_{i}'] = val
    
    return BenchmarkResult(
        name=f'GraphColor({num_nodes}n,{len(edges)}e,{num_colors}c)',
        method='ddb',
        time_seconds=elapsed,
        contradictions=contradictions,
        solution_found=len(solution) == num_nodes,
        solution=solution,
    )


def run_graph_coloring_cdcl(num_nodes: int = 10, edge_prob: float = 0.3, num_colors: int = 3, seed: int = 42) -> BenchmarkResult:
    """Run graph coloring with CDCL."""
    import random
    from propagator import Cell, initialize_scheduler, enable_cdcl, get_contradictions, get_cdcl_engine
    from propagator.primitives import eq
    from propagator.guessing_machine import one_of, abhor
    from propagator.tms import tms_query, tms_p
    from propagator.scheduler import run
    
    random.seed(seed)
    
    # Generate random graph
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                edges.append((i, j))
    
    initialize_scheduler()
    engine = enable_cdcl()
    
    start = time.time()
    
    # Node color cells
    nodes = [Cell(name=f'node_{i}') for i in range(num_nodes)]
    colors = list(range(num_colors))
    
    # Each node has a color
    for node in nodes:
        one_of(colors, node)
    
    # Adjacent nodes have different colors
    for i, j in edges:
        eq_cell = Cell(name=f'eq_{i}_{j}')
        eq(nodes[i], nodes[j], eq_cell)
        abhor(eq_cell)
    
    run()
    
    elapsed = time.time() - start
    contradictions = len(get_contradictions())
    
    # Extract solution
    solution = {}
    for i, node in enumerate(nodes):
        content = node.content
        if tms_p(content):
            result = tms_query(content)
            if result is not None:
                val = result.value if hasattr(result, 'value') else result
                solution[f'node_{i}'] = val
    
    return BenchmarkResult(
        name=f'GraphColor({num_nodes}n,{len(edges)}e,{num_colors}c)',
        method='cdcl',
        time_seconds=elapsed,
        contradictions=contradictions,
        solution_found=len(solution) == num_nodes,
        solution=solution,
        cdcl_decisions=engine.stats.decisions,
        cdcl_backjumps=engine.stats.backjumps,
        cdcl_levels_saved=engine.stats.backjump_levels_saved,
    )


def run_benchmark_suite(verbose: bool = True):
    """Run the full benchmark suite."""
    results = []
    
    print("=" * 70)
    print("CDCL vs DDB Performance Comparison")
    print("=" * 70)
    print()
    
    # Multiple Dwelling
    print("Running: Multiple Dwelling Puzzle")
    ddb_result = run_multiple_dwelling_ddb()
    cdcl_result = run_multiple_dwelling_cdcl()
    results.extend([ddb_result, cdcl_result])
    print(ddb_result.summary())
    print(cdcl_result.summary())
    
    # Verify same solution
    if ddb_result.solution == cdcl_result.solution:
        print(f"  ✓ Solutions match: {ddb_result.solution}")
    else:
        print(f"  ✗ Solutions differ!")
        print(f"    DDB:  {ddb_result.solution}")
        print(f"    CDCL: {cdcl_result.solution}")
    print()
    
    # N-Queens (small)
    for n in [4, 5, 6]:
        print(f"Running: {n}-Queens")
        ddb_result = run_n_queens_ddb(n)
        cdcl_result = run_n_queens_cdcl(n)
        results.extend([ddb_result, cdcl_result])
        print(ddb_result.summary())
        print(cdcl_result.summary())
        print()
    
    # Graph Coloring
    for num_nodes, edge_prob, num_colors in [(8, 0.4, 3), (10, 0.3, 3), (12, 0.25, 3)]:
        print(f"Running: Graph Coloring ({num_nodes} nodes)")
        ddb_result = run_graph_coloring_ddb(num_nodes, edge_prob, num_colors)
        cdcl_result = run_graph_coloring_cdcl(num_nodes, edge_prob, num_colors)
        results.extend([ddb_result, cdcl_result])
        print(ddb_result.summary())
        print(cdcl_result.summary())
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Group by problem
    problems = {}
    for r in results:
        if r.name not in problems:
            problems[r.name] = {}
        problems[r.name][r.method] = r
    
    print(f"{'Problem':<35} {'DDB':<12} {'CDCL':<12} {'Speedup':<10} {'Conflicts':}")
    print("-" * 70)
    
    total_ddb_time = 0
    total_cdcl_time = 0
    
    for name, methods in problems.items():
        ddb = methods.get('ddb')
        cdcl = methods.get('cdcl')
        
        if ddb and cdcl:
            speedup = ddb.time_seconds / cdcl.time_seconds if cdcl.time_seconds > 0 else float('inf')
            conflict_ratio = f"{ddb.contradictions}/{cdcl.contradictions}"
            print(f"{name:<35} {ddb.time_seconds:<12.4f} {cdcl.time_seconds:<12.4f} {speedup:<10.2f}x {conflict_ratio}")
            total_ddb_time += ddb.time_seconds
            total_cdcl_time += cdcl.time_seconds
    
    print("-" * 70)
    total_speedup = total_ddb_time / total_cdcl_time if total_cdcl_time > 0 else float('inf')
    print(f"{'TOTAL':<35} {total_ddb_time:<12.4f} {total_cdcl_time:<12.4f} {total_speedup:<10.2f}x")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='CDCL vs DDB benchmark')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--problem', '-p', choices=['dwelling', 'queens', 'coloring', 'all'],
                        default='all', help='Which problem to run')
    args = parser.parse_args()
    
    if args.problem == 'all':
        run_benchmark_suite(verbose=args.verbose)
    elif args.problem == 'dwelling':
        print("Multiple Dwelling - DDB:")
        ddb = run_multiple_dwelling_ddb()
        print(ddb.summary())
        print(f"  Solution: {ddb.solution}")
        print()
        print("Multiple Dwelling - CDCL:")
        cdcl = run_multiple_dwelling_cdcl()
        print(cdcl.summary())
        print(f"  Solution: {cdcl.solution}")
    elif args.problem == 'queens':
        for n in [4, 5, 6, 7]:
            print(f"{n}-Queens - DDB:")
            ddb = run_n_queens_ddb(n)
            print(ddb.summary())
            print(f"{n}-Queens - CDCL:")
            cdcl = run_n_queens_cdcl(n)
            print(cdcl.summary())
            print()
    elif args.problem == 'coloring':
        print("Graph Coloring - DDB:")
        ddb = run_graph_coloring_ddb(10, 0.3, 3)
        print(ddb.summary())
        print("Graph Coloring - CDCL:")
        cdcl = run_graph_coloring_cdcl(10, 0.3, 3)
        print(cdcl.summary())


if __name__ == '__main__':
    main()
