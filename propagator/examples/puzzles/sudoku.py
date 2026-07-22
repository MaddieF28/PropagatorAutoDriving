"""
Sudoku solver using propagator networks.

This implementation follows the Scheme implementation in propagator/examples/sudoku.scm
It uses:
- Cells for each position on the board
- all-different constraints (via require_distinct) for rows, cols, and squares
- Guessing (via one_of) to explore possibilities
- TMS (Truth Maintenance System) for backtracking when contradictions arise

IMPORTANT: Performance Considerations
=====================================
The propagator-based Sudoku solver has inherently different performance
characteristics compared to traditional backtracking CSP solvers:

1. The Scheme implementation (~2009) took ~3-5 minutes for a medium puzzle
   with 629 backtracking failures. The Python implementation is similar.

2. For each unknown cell, `one_of` creates O(n) cells and propagators for
   an n-value domain. For 9x9 Sudoku with ~50 unknowns, this creates
   thousands of propagators.

3. The TMS backtracking mechanism has overhead from tracking dependency
   support sets and premise nogoods.

For practical Sudoku solving, use the CSP implementation in csp/ which uses
traditional backtracking with constraint propagation (AC-3, MAC, etc.).

Missing Features in Python Propagator Implementation
=====================================================
Compared to the Scheme implementation, the Python version has:

1. PRESENT: TMS (Truth Maintenance System) for dependency tracking
2. PRESENT: binary_amb for binary choice/search
3. PRESENT: one_of for n-ary choice
4. PRESENT: require_distinct (all-different constraint via pairwise inequality)
5. PRESENT: require/abhor for asserting truth values

The core propagator infrastructure appears complete for Sudoku solving.
The main difference from Scheme is Python's performance characteristics.

Usage:
    python -m propagator.examples.puzzles.sudoku          # Run 4x4 puzzle
    python -m propagator.examples.puzzles.sudoku --9x9    # Include 9x9 trivial
    python -m propagator.examples.puzzles.sudoku --hard   # Try Scheme's example (slow)
"""

from typing import List, Optional, Union
import math
import time

from propagator import (
    Cell,
    initialize_scheduler,
    run,
    constant,
    set_auto_run,
    get_auto_run,
)
from propagator.guessing_machine import require_distinct, one_of
from propagator.tms import (
    tms_query,
    tms_p,
    number_of_calls_to_fail,
    get_contradictions,
)
from propagator.nothing import nothing_p
from propagator.supported_values import Supported
from propagator.cdcl import enable_cdcl


class SudokuBoard:
    """
    Represents a Sudoku board with cells organized in a grid.
    
    Scheme equivalent:
        (define-structure sudoku-board rt-size cells)
    """
    
    def __init__(self, rt_size: int):
        """
        Create an empty Sudoku board.
        
        Args:
            rt_size: The square root of the board size (e.g., 3 for 9x9 board)
        """
        self.rt_size = rt_size
        self.size = rt_size * rt_size
        
        # Create grid of cells
        # Scheme: (make-initialized-vector size (lambda (row) ...))
        self.cells: List[List[Cell]] = []
        for row in range(self.size):
            row_cells = []
            for col in range(self.size):
                cell = Cell(name=f"cell[{row},{col}]")
                row_cells.append(cell)
            self.cells.append(row_cells)
    
    def ref(self, row: int, col: int) -> Cell:
        """
        Get the cell at the specified position.
        
        Scheme equivalent:
            (define (sudoku-board-ref board row col) ...)
        """
        return self.cells[row][col]
    
    def get_all_cells(self) -> List[Cell]:
        """
        Get all cells in the board as a flat list.
        
        Scheme equivalent:
            (define (sudoku-board-cells board)
              (apply append (sudoku-board-rows board)))
        """
        return [cell for row in self.cells for cell in row]
    
    def get_rows(self) -> List[List[Cell]]:
        """
        Get all rows as lists of cells.
        
        Scheme equivalent:
            (define (sudoku-board-rows board)
              (map vector->list (vector->list (sudoku-board-cells board))))
        """
        return self.cells
    
    def get_cols(self) -> List[List[Cell]]:
        """
        Get all columns as lists of cells.
        
        Scheme equivalent:
            (define (sudoku-board-cols board)
              (map (lambda (col) ...) (iota (sudoku-board-size board))))
        """
        cols = []
        for col in range(self.size):
            col_cells = [self.cells[row][col] for row in range(self.size)]
            cols.append(col_cells)
        return cols
    
    def get_square_at(self, start_row: int, start_col: int) -> List[Cell]:
        """
        Get cells in the square starting at the given position.
        
        Scheme equivalent:
            (define (sudoku-board-square-at board row col) ...)
        """
        cells = []
        for drow in range(self.rt_size):
            for dcol in range(self.rt_size):
                cells.append(self.cells[start_row + drow][start_col + dcol])
        return cells
    
    def get_squares(self) -> List[List[Cell]]:
        """
        Get all square blocks as lists of cells.
        
        Scheme equivalent:
            (define (sudoku-board-squares board) ...)
        """
        squares = []
        for block_row in range(self.rt_size):
            for block_col in range(self.rt_size):
                start_row = block_row * self.rt_size
                start_col = block_col * self.rt_size
                squares.append(self.get_square_at(start_row, start_col))
        return squares


def add_different_constraints(board: SudokuBoard) -> None:
    """
    Add all-different constraints for rows, columns, and squares.
    
    Scheme equivalent:
        (define (add-different-constraints! board)
          (for-each (lambda (shape)
                      (for-each (lambda (cells)
                                  (apply all-different cells))
                                (shape board)))
                    (list sudoku-board-rows sudoku-board-cols sudoku-board-squares)))
    """
    # Add constraints for each row
    for row_cells in board.get_rows():
        require_distinct(row_cells)
    
    # Add constraints for each column
    for col_cells in board.get_cols():
        require_distinct(col_cells)
    
    # Add constraints for each square
    for square_cells in board.get_squares():
        require_distinct(square_cells)


def add_known_values(board: SudokuBoard, puzzle: List[List[int]]) -> None:
    """
    Add known values from the puzzle specification.
    
    Values of 0 represent unknown cells.
    
    Scheme equivalent:
        (define (add-known-values! board board-by-rows)
          (for-each (lambda (board-row spec-row)
                      (for-each (lambda (board-cell spec)
                                  (if (and (integer? spec)
                                           (<= 1 spec 9))
                                      (add-content board-cell (one-choice spec))))
                                board-row spec-row))
                    (sudoku-board-rows board) board-by-rows))
    """
    for row_idx, (board_row, spec_row) in enumerate(zip(board.get_rows(), puzzle)):
        for col_idx, (cell, value) in enumerate(zip(board_row, spec_row)):
            if isinstance(value, int) and 1 <= value <= board.size:
                # Known value - add as constant
                constant(value, cell)


def add_guessers(board: SudokuBoard) -> None:
    """
    Add guessers to cells that don't already have content.
    
    This allows the search mechanism to try different values.
    
    Scheme equivalent:
        (define (add-guessers! board)
          (for-each (lambda (row)
                      (for-each (lambda (cell)
                                  (add-guesser! cell (sudoku-board-size board)))
                                row))
                    (sudoku-board-rows board)))
        
        (define (add-guesser! cell size)
          (if (not (integer? (content cell)))
              (apply one-of `(,@(iota size 1) ,cell))))
    """
    possible_values = list(range(1, board.size + 1))
    
    for row in board.get_rows():
        for cell in row:
            # Only add guesser if cell doesn't already have a definite integer value
            # This matches the Scheme: (if (not (integer? (content cell))) ...)
            if not isinstance(cell.content, int):
                one_of(possible_values, cell)


def is_one_choice(thing) -> bool:
    """
    Check if the value represents a definite choice.
    
    Scheme equivalent:
        (define (one-choice? thing)
          (or (integer? thing)
              (and (tms? thing)
                   (not (nothing? (tms-query thing))))))
    """
    if isinstance(thing, int):
        return True
    if tms_p(thing):
        query_result = tms_query(thing)
        return not nothing_p(query_result)
    return False


def get_the_one_choice(thing) -> Optional[int]:
    """
    Extract the definite value from a cell content.
    
    Scheme equivalent:
        (define (the-one-choice thing)
          (if (integer? thing)
              thing
              (v&s-value (tms-query thing))))
    """
    if isinstance(thing, int):
        return thing
    if tms_p(thing):
        query_result = tms_query(thing)
        if nothing_p(query_result):
            return None
        if isinstance(query_result, Supported):
            return query_result.value
        return query_result
    return None


def parse_sudoku(puzzle: List[List[int]]) -> SudokuBoard:
    """
    Parse a puzzle specification and create a Sudoku board with constraints.
    
    Scheme equivalent:
        (define (parse-sudoku board-by-rows)
          (let* ((rt-size (inexact->exact (sqrt (length board-by-rows))))
                 (board (empty-sudoku-board rt-size)))
            (add-different-constraints! board)
            (add-known-values! board board-by-rows)
            (add-guessers! board)
            board))
    """
    size = len(puzzle)
    rt_size = int(math.sqrt(size))
    
    if rt_size * rt_size != size:
        raise ValueError(f"Puzzle size {size} is not a perfect square")
    
    board = SudokuBoard(rt_size)
    
    # Add constraints in the same order as Scheme
    add_different_constraints(board)
    add_known_values(board, puzzle)
    add_guessers(board)
    
    return board


def print_sudoku_board(board: SudokuBoard, show_separators: bool = True) -> None:
    """
    Print the Sudoku board.
    
    Scheme equivalent:
        (define (print-sudoku-board board)
          (for-each (lambda (row)
                      (for-each (lambda (cell)
                                  (if (one-choice? (content cell))
                                      (display (the-one-choice (content cell)))
                                      (display "?")))
                                row)
                      (newline))
                    (sudoku-board-rows board))
          board)
    """
    for row_idx, row in enumerate(board.get_rows()):
        if show_separators and row_idx > 0 and row_idx % board.rt_size == 0:
            print("-" * (board.size + board.rt_size - 1))
        
        row_str = ""
        for col_idx, cell in enumerate(row):
            if show_separators and col_idx > 0 and col_idx % board.rt_size == 0:
                row_str += "|"
            
            if is_one_choice(cell.content):
                value = get_the_one_choice(cell.content)
                row_str += str(value) if value else "?"
            else:
                row_str += "?"
        print(row_str)


def do_sudoku(puzzle: List[List[int]], verbose: bool = True) -> SudokuBoard:
    """
    Solve a Sudoku puzzle using propagator networks.
    
    Scheme equivalent:
        (define (do-sudoku board-by-rows)
          (initialize-scheduler)
          (let ((board (parse-sudoku board-by-rows)))
            (run)
            (print-sudoku-board board)))
    """
    initialize_scheduler()
    engine = enable_cdcl()

    
    # Disable auto-run during construction to avoid running propagation
    # prematurely while building the network
    old_auto_run = get_auto_run()
    set_auto_run(False)
    
    try:
        if verbose:
            print("Parsing puzzle and creating constraints...")
        
        board = parse_sudoku(puzzle)
        
        if verbose:
            print("Running propagator network...")
        
        # Now run the full propagation
        set_auto_run(True)
        start_time = time.time()
        run()
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"\nSolved in {elapsed:.3f} seconds")
            print_sudoku_board(board)
        
        return board
    finally:
        # Restore original auto-run setting
        set_auto_run(old_auto_run)


def count_failures(func):
    """
    Decorator to count failures during puzzle solving.
    
    Scheme equivalent:
        (define (count-failures thunk)
          (fluid-let ((*number-of-calls-to-fail* 0))
            (let ((value (thunk)))
              (pp `(failed ,*number-of-calls-to-fail* times))
              value)))
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        contradictions = get_contradictions()
        print(f"Failed {len(contradictions)} times")
        return result
    return wrapper


def verify_solution(board: SudokuBoard) -> bool:
    """
    Verify that the solution is complete and valid.
    """
    # Check all cells have values
    for cell in board.get_all_cells():
        if not is_one_choice(cell.content):
            return False
    
    # Get all values
    def get_values(cells: List[Cell]) -> List[int]:
        return [get_the_one_choice(cell.content) for cell in cells]
    
    expected = set(range(1, board.size + 1))
    
    # Check rows
    for row in board.get_rows():
        if set(get_values(row)) != expected:
            return False
    
    # Check columns
    for col in board.get_cols():
        if set(get_values(col)) != expected:
            return False
    
    # Check squares
    for square in board.get_squares():
        if set(get_values(square)) != expected:
            return False
    
    return True


# Example puzzles from the Scheme implementation

PUZZLE_EASY = [
    [0, 0, 7, 0, 0, 0, 6, 5, 0],
    [8, 4, 6, 0, 0, 5, 1, 0, 9],
    [0, 0, 9, 0, 0, 0, 0, 0, 3],
    [1, 0, 0, 5, 6, 0, 0, 9, 4],
    [0, 0, 0, 9, 4, 8, 0, 0, 0],
    [4, 9, 0, 0, 1, 2, 0, 0, 5],
    [7, 0, 0, 0, 0, 0, 9, 0, 0],
    [9, 0, 5, 2, 0, 0, 4, 1, 7],
    [0, 3, 1, 0, 0, 0, 5, 0, 0],
]

# This puzzle is harder and may take longer / require more backtracking
PUZZLE_HARD = [
    [0, 0, 8, 0, 1, 0, 0, 4, 0],
    [0, 4, 1, 6, 0, 0, 7, 8, 0],
    [0, 0, 6, 0, 7, 8, 0, 0, 0],
    [0, 0, 0, 7, 0, 0, 9, 3, 0],
    [0, 9, 0, 0, 0, 0, 0, 5, 0],
    [0, 2, 3, 0, 0, 5, 0, 0, 0],
    [0, 0, 0, 9, 5, 0, 8, 0, 0],
    [0, 8, 9, 0, 0, 4, 5, 7, 0],
    [0, 7, 0, 0, 8, 0, 1, 0, 0],
]

# Simple 4x4 puzzle for testing
PUZZLE_4X4 = [
    [1, 0, 0, 4],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [4, 0, 0, 1],
]

# A very simple 9x9 puzzle with many givens (minimal backtracking needed)
# This is good for testing that the implementation works
PUZZLE_TRIVIAL_9X9 = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]


if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("SUDOKU SOLVER USING PROPAGATOR NETWORKS")
    print("=" * 50)
    print()
    print("NOTE: This implementation uses propagator networks with TMS")
    print("(Truth Maintenance System) for backtracking search.")
    print("Performance is similar to the original Scheme implementation:")
    print("- 4x4 puzzles: ~1-2 seconds")
    print("- 9x9 easy puzzles: minutes to hours (many backtracking steps)")
    print()
    print("For fast Sudoku solving, see the CSP implementation instead:")
    print("  python3 demo_csp.py")
    print()
    
    # Check for command line arguments
    run_hard = "--hard" in sys.argv or "-h" in sys.argv
    run_9x9 = "--9x9" in sys.argv or run_hard
    
    # Start with a simple 4x4 puzzle
    print("--- 4x4 Puzzle ---")
    print("Input:")
    for row in PUZZLE_4X4:
        print("".join(str(x) if x else "." for x in row))
    print("\nSolving...")
    
    board = do_sudoku(PUZZLE_4X4)
    
    if verify_solution(board):
        print("✓ Solution verified!")
    else:
        print("✗ Solution incomplete or invalid")
    
    if run_9x9:
        # Try the 9x9 puzzle
        print("\n" + "=" * 50)
        puzzle_name = "9x9 Easy Puzzle (from Scheme)" if run_hard else "9x9 Trivial Puzzle"
        puzzle_to_use = PUZZLE_EASY if run_hard else PUZZLE_TRIVIAL_9X9
        
        print(f"\n--- {puzzle_name} ---")
        print("Input:")
        for row in puzzle_to_use:
            print("".join(str(x) if x else "." for x in row))
        print("\nSolving (this may take a while)...")
        
        board = do_sudoku(puzzle_to_use)
        
        contradictions = get_contradictions()
        print(f"\nContradictions encountered: {len(contradictions)}")
        
        if verify_solution(board):
            print("✓ Solution verified!")
        else:
            print("✗ Solution incomplete or invalid")
    else:
        print("\n" + "-" * 50)
        print("Skipping 9x9 puzzles (can take minutes).")
        print("Run with --9x9 to try the trivial 9x9 puzzle.")
        print("Run with --hard to try the original Scheme puzzle.")
