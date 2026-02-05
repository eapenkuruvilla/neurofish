#!/usr/bin/env python3
"""
Parameter tuning script for Neurofish chess engine using engine_test.

Usage:
    ./engine_test_batch.py PARAM_NAME value1 value2 value3 ...
    ./engine_test_batch.py PARAM_NAME value1 value2 --mp

Examples:
    ./engine_test_batch.py QS_SOFT_STOP_DIVISOR 7.0 8.0 9.0 10.0
    ./engine_test_batch.py MAX_QS_MOVES "[12,6,4,2]" "[10,5,3,2]" "[15,8,5,3]"
    ./engine_test_batch.py FUTILITY_MAX_DEPTH 2 3 4 --mp

Note: The parameter is passed as an environment variable to engine_test.py.
      Make sure your config.py reads the parameter from os.environ.
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
    success_rate: float
    passed: int
    total: int
    time_avg: float
    time_max: float


def parse_engine_test_output(output: str) -> Optional[TuneResult]:
    """Parse the output from engine_test.py to extract success rate and timing stats."""

    # Extract ALL success rate lines and take the LAST one (final result)
    # Example: "total=300, passed=231, success-rate=77.0%"
    success_matches = re.findall(
        r'total=(\d+),\s*passed=(\d+),\s*success-rate=([\d.]+)%',
        output
    )

    if not success_matches:
        return None

    # Take the last match (final result)
    last_match = success_matches[-1]
    total = int(last_match[0])
    passed = int(last_match[1])
    success_rate = float(last_match[2])

    # Extract timing stats
    # Example: "time-avg=0.63, time-max=2.84"
    time_match = re.search(
        r'time-avg=([\d.]+),\s*time-max=([\d.]+)',
        output
    )

    if time_match:
        time_avg = float(time_match.group(1))
        time_max = float(time_match.group(2))
    else:
        time_avg = 0.0
        time_max = 0.0

    return TuneResult(
        param_value="",  # Will be filled in by caller
        success_rate=success_rate,
        passed=passed,
        total=total,
        time_avg=time_avg,
        time_max=time_max
    )


def run_tuning(param_name: str, param_value: str, mp: bool) -> Optional[TuneResult]:
    """Run engine_test.py with a specific parameter value."""

    # Set up environment with the parameter override
    env = os.environ.copy()
    env[param_name] = param_value

    # Build command
    cmd = ["python", "-m", "test.engine_test"]
    if mp:
        cmd.append("--mp")

    # Build the command string for display
    cmd_display = f"{param_name}={param_value} {' '.join(cmd)}"

    print(f"\n{'=' * 70}")
    print(f"Testing {param_name}={param_value}")
    print(f"{'=' * 70}")
    print(f"  {cmd_display}")
    print(f"\n{'-' * 70}\n")

    try:
        # Force unbuffered output from the child Python process
        env['PYTHONUNBUFFERED'] = '1'

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output_lines = []
        last_total = 0
        started_tests = False

        for line in process.stdout:
            output_lines.append(line)
            line_stripped = line.strip()

            # Check if this is a progress line (total=X, passed=Y, success-rate=Z%)
            progress_match = re.search(
                r'total=(\d+),\s*passed=(\d+),\s*success-rate=([\d.]+)%',
                line
            )

            if progress_match:
                started_tests = True
                total = int(progress_match.group(1))
                passed = int(progress_match.group(2))
                rate = progress_match.group(3)

                # Check if the previous line was a failure message
                if len(output_lines) >= 2:
                    prev_line = output_lines[-2].strip()
                    if prev_line.startswith("Failed test:"):
                        # Extract FEN from failure line
                        fen_match = re.search(r'fen=([^,]+)', prev_line)
                        if fen_match:
                            fen = fen_match.group(1)
                            # Clear the line and print failure on new line
                            sys.stdout.write(f"\r\033[K#{total} FAIL: {fen[:50]}{'...' if len(fen) > 50 else ''}\n")
                            sys.stdout.flush()

                # Always update progress line (use ANSI escape to clear line)
                sys.stdout.write(f"\rProgress: {total} tests, {passed} passed ({rate}%)\033[K")
                sys.stdout.flush()

                last_total = total

            elif "time-avg=" in line:
                # Final timing line - print on new line
                sys.stdout.write(f"\n{line_stripped}\n")
                sys.stdout.flush()

            elif not started_tests:
                # Print all initialization output before tests start
                sys.stdout.write(line)
                sys.stdout.flush()

            elif "CONFIGURATION OVERRIDES" in line:
                # Config section header (appears at end)
                sys.stdout.write(f"{line_stripped}\n")
                sys.stdout.flush()
            elif "->" in line_stripped and ":" in line_stripped:
                # Config override line (appears at end)
                sys.stdout.write(f"{line_stripped}\n")
                sys.stdout.flush()

        # Clear the progress line
        print()

        process.wait()
        output = ''.join(output_lines)

        if process.returncode != 0:
            print(f"\nWarning: engine_test exited with code {process.returncode}")

        result = parse_engine_test_output(output)
        if result:
            result.param_value = param_value
        return result

    except FileNotFoundError:
        print(f"Error: Could not find python or test.engine_test module")
        return None
    except Exception as e:
        print(f"Error running tuning: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_summary(param_name: str, results: List[TuneResult]):
    """Print a summary table of all results."""

    if not results:
        print("\nNo results to summarize.")
        return

    print(f"\n{'=' * 90}")
    print(f"TUNING SUMMARY: {param_name}")
    print(f"{'=' * 90}")

    # Header
    print(f"\n{'Value':<30} {'Success%':>10} {'Passed':>10} {'Total':>8} {'TimeAvg':>10} {'TimeMax':>10}")
    print("-" * 90)

    # Sort by success rate (descending)
    sorted_results = sorted(results, key=lambda r: r.success_rate, reverse=True)

    for r in sorted_results:
        print(
            f"{r.param_value:<30} {r.success_rate:>10.2f} {r.passed:>10} {r.total:>8} {r.time_avg:>10.2f} {r.time_max:>10.2f}")

    print("-" * 90)

    # Best result
    best = sorted_results[0]
    print(f"\n*** BEST VALUE: {param_name}={best.param_value}")
    print(f"    Success Rate: {best.success_rate:.2f}% ({best.passed}/{best.total})")
    print(f"    Time: avg={best.time_avg:.2f}s, max={best.time_max:.2f}s")

    # If there are multiple results, show comparison
    if len(sorted_results) > 1:
        worst = sorted_results[-1]
        diff = best.success_rate - worst.success_rate
        print(f"\n    Improvement over worst ({worst.param_value}): {diff:+.2f}%")

    print(f"\n{'=' * 90}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Tune a single parameter by testing multiple values with engine_test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s QS_SOFT_STOP_DIVISOR 7.0 8.0 9.0 10.0
    %(prog)s MAX_QS_MOVES "[12,6,4,2]" "[10,5,3,2]" "[15,8,5,3]"
    %(prog)s FUTILITY_MAX_DEPTH 2 3 4 --mp
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
        "--mp",
        action="store_true",
        help="Enable multi-processing mode"
    )

    args = parser.parse_args()

    print(f"\n{'#' * 80}")
    print(f"# PARAMETER TUNING")
    print(f"# Parameter: {args.param_name}")
    print(f"# Values to test: {args.values}")
    print(f"# Multi-processing: {args.mp}")
    print(f"{'#' * 80}")

    results = []

    for i, value in enumerate(args.values, 1):
        print(f"\n[{i}/{len(args.values)}] Testing value: {value}")

        result = run_tuning(
            param_name=args.param_name,
            param_value=value,
            mp=args.mp
        )

        if result:
            results.append(result)
            print(f"\n>>> Result for {args.param_name}={value}: "
                  f"{result.success_rate:.2f}% ({result.passed}/{result.total})")
        else:
            print(f"\n>>> Failed to get result for {value}")

    # Print final summary
    print_summary(args.param_name, results)

    return 0


if __name__ == "__main__":
    sys.exit(main())