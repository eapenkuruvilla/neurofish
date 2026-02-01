#!/usr/bin/env python3
"""
Parameter tuning script for Neurofish chess engine.

Usage:
    ./stockfish_batch.py PARAM_NAME value1 value2 value3 ...
    ./stockfish_batch.py PARAM_NAME value1 value2 --games 50 --elo 2000

Examples:
    ./stockfish_batch.py QS_SOFT_STOP_DIVISOR 7.0 8.0 9.0 10.0
    ./stockfish_batch.py MAX_QS_MOVES "[12,6,4,2]" "[10,5,3,2]" "[15,8,5,3]"
    ./stockfish_batch.py FUTILITY_MAX_DEPTH 2 3 4 --games 50 --elo 2400
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TuneResult:
    """Result from a single tuning run."""
    param_value: str
    elo_diff: float
    elo_error: float
    los: float
    draw_ratio: float
    wins: int
    losses: int
    draws: int
    games: int


def parse_stockfish_output(output: str) -> Optional[TuneResult]:
    """Parse the output from stockfish.sh to extract Elo and win/loss stats."""

    # Check for engine initialization failure
    if "Could not initialize player" in output or "Terminating process of engine" in output:
        print("\n*** ENGINE INITIALIZATION FAILED ***")
        print("Check that the engine starts correctly outside of cutechess.")
        return None

    # Extract Elo difference line
    # Example: "Elo difference: 94.9 +/- 125.5, LOS: 94.2 %, DrawRatio: 13.3 %"
    elo_match = re.search(
        r'Elo difference:\s*([-\d.]+|nan)\s*\+/-\s*([\d.]+|nan),\s*LOS:\s*([\d.]+|nan)\s*%,\s*DrawRatio:\s*([\d.]+|nan)\s*%',
        output
    )

    if not elo_match:
        return None

    # Check for nan values (happens when no games complete)
    elo_str = elo_match.group(1)
    if elo_str == 'nan':
        print("\n*** NO VALID GAMES COMPLETED ***")
        print("Elo difference is 'nan' - check engine logs for errors.")
        return None

    elo_diff = float(elo_str)
    elo_error = float(elo_match.group(2)) if elo_match.group(2) != 'nan' else 0.0
    los = float(elo_match.group(3)) if elo_match.group(3) != 'nan' else 0.0
    draw_ratio = float(elo_match.group(4)) if elo_match.group(4) != 'nan' else 0.0

    # Extract final score line
    # Example: "Score of Neurofish vs stockfish2200: 17 - 9 - 4  [0.633] 30"
    score_match = re.search(
        r'Score of Neurofish vs stockfish\d+:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)',
        output
    )

    if score_match:
        wins = int(score_match.group(1))
        losses = int(score_match.group(2))
        draws = int(score_match.group(3))
        games = wins + losses + draws
    else:
        wins = losses = draws = games = 0

    # If no games were played, return None
    if games == 0:
        print("\n*** NO GAMES WERE PLAYED ***")
        return None

    return TuneResult(
        param_value="",  # Will be filled in by caller
        elo_diff=elo_diff,
        elo_error=elo_error,
        los=los,
        draw_ratio=draw_ratio,
        wins=wins,
        losses=losses,
        draws=draws,
        games=games
    )


def run_tuning(param_name: str, param_value: str, games: int, elo: int,
               script_path: str) -> Optional[TuneResult]:
    """Run stockfish.sh with a specific parameter value."""

    # Set up environment with the parameter override
    env = os.environ.copy()
    env[param_name] = param_value

    # Build command
    cmd = [script_path, "Neurofish", str(elo), str(games)]

    # Build the command string for display (can be copy/pasted to terminal)
    cmd_display = f"{param_name}={param_value} {' '.join(cmd)}"

    print(f"\n{'=' * 70}")
    print(f"Testing {param_name}={param_value}")
    print(f"{'=' * 70}")
    print(f"  {cmd_display}")
    print(f"\n{'-' * 70}\n")

    try:
        # Run the script and let output go to console (stdout/stderr not captured)
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        # Collect output while also printing to console
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Print to console in real-time
            output_lines.append(line)

        process.wait()
        output = ''.join(output_lines)

        if process.returncode != 0:
            print(f"\nWarning: stockfish.sh exited with code {process.returncode}")

        result = parse_stockfish_output(output)
        if result:
            result.param_value = param_value
        return result

    except FileNotFoundError:
        print(f"Error: Could not find script at {script_path}")
        return None
    except Exception as e:
        print(f"Error running tuning: {e}")
        return None


def print_summary(param_name: str, results: List[TuneResult], stockfish_elo: int):
    """Print a summary table of all results."""

    if not results:
        print("\nNo results to summarize.")
        return

    print(f"\n{'=' * 90}")
    print(f"TUNING SUMMARY: {param_name}")
    print(f"{'=' * 90}")

    # Header
    print(f"\n{'Value':<30} {'Est.Elo':>10} {'EloDiff':>10} {'±Error':>10} {'W-L-D':>12} {'LOS%':>8}")
    print("-" * 90)

    # Sort by Elo difference (descending)
    sorted_results = sorted(results, key=lambda r: r.elo_diff, reverse=True)

    for r in sorted_results:
        wld = f"{r.wins}-{r.losses}-{r.draws}"
        est_elo = stockfish_elo + r.elo_diff
        print(f"{r.param_value:<30} {est_elo:>10.0f} {r.elo_diff:>+10.1f} {r.elo_error:>10.1f} {wld:>12} {r.los:>8.1f}")

    print("-" * 90)

    # Best result
    best = sorted_results[0]
    best_est_elo = stockfish_elo + best.elo_diff
    print(f"\n*** BEST VALUE: {param_name}={best.param_value}")
    print(f"    Estimated Elo: {best_est_elo:.0f} (SF {stockfish_elo} + {best.elo_diff:+.1f} ±{best.elo_error:.1f})")
    print(f"    W-L-D: {best.wins}-{best.losses}-{best.draws}, LOS: {best.los:.1f}%")

    # If there are multiple results, show comparison
    if len(sorted_results) > 1:
        worst = sorted_results[-1]
        diff = best.elo_diff - worst.elo_diff
        print(f"\n    Improvement over worst ({worst.param_value}): {diff:+.1f} Elo")

    print(f"\n{'=' * 90}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Tune a single parameter by testing multiple values against Stockfish.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s QS_SOFT_STOP_DIVISOR 7.0 8.0 9.0 10.0
    %(prog)s MAX_QS_MOVES "[12,6,4,2]" "[10,5,3,2]" "[15,8,5,3]"
    %(prog)s FUTILITY_MAX_DEPTH 2 3 4 --games 50 --elo 2400
        """
    )

    parser.add_argument(
        "param_name",
        help="Name of the parameter to tune (e.g., QS_SOFT_STOP_DIVISOR)"
    )

    parser.add_argument(
        "values",
        nargs="+",
        help="Values to test (use quotes for lists, e.g., \"[12,6,4,2]\")"
    )

    parser.add_argument(
        "--games", "-g",
        type=int,
        default=30,
        help="Number of games per value (default: 30)"
    )

    parser.add_argument(
        "--elo", "-e",
        type=int,
        default=2200,
        help="Stockfish ELO level (default: 2200)"
    )

    parser.add_argument(
        "--script", "-s",
        default=None,
        help="Path to stockfish.sh (default: auto-detect from script location)"
    )

    args = parser.parse_args()

    # Find stockfish.sh script
    if args.script:
        script_path = args.script
    else:
        # Try to find it relative to this script or in common locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, "stockfish.sh"),
            os.path.join(script_dir, "test", "stockfish.sh"),
            os.path.join(script_dir, "..", "test", "stockfish.sh"),
            "test/stockfish.sh",
            "./stockfish.sh",
        ]

        script_path = None
        for path in possible_paths:
            if os.path.isfile(path):
                script_path = path
                break

        if not script_path:
            print("Error: Could not find stockfish.sh")
            print("Please specify path with --script option")
            sys.exit(1)

    if not os.path.isfile(script_path):
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)

    print(f"\n{'#' * 80}")
    print(f"# PARAMETER TUNING")
    print(f"# Parameter: {args.param_name}")
    print(f"# Values to test: {args.values}")
    print(f"# Games per value: {args.games}")
    print(f"# Stockfish ELO: {args.elo}")
    print(f"# Script: {script_path}")
    print(f"{'#' * 80}")

    results = []

    for i, value in enumerate(args.values, 1):
        print(f"\n[{i}/{len(args.values)}] Testing value: {value}")

        result = run_tuning(
            param_name=args.param_name,
            param_value=value,
            games=args.games,
            elo=args.elo,
            script_path=script_path
        )

        if result:
            results.append(result)
            print(f"\n>>> Result for {args.param_name}={value}: "
                  f"Elo {result.elo_diff:+.1f} ±{result.elo_error:.1f}")
        else:
            print(f"\n>>> Failed to get result for {value}")

    # Print final summary
    print_summary(args.param_name, results, args.elo)

    return 0


if __name__ == "__main__":
    sys.exit(main())