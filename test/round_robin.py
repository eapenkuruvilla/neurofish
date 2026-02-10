#!/usr/bin/env python3
"""
tune.py Ã¢â‚¬â€ Self-play parameter tuning for Neurofish chess engine.

Runs a round-robin tournament via cutechess-cli where each participant
is the same engine configured with a different value of one UCI parameter.
Estimates relative Elo ratings using Bradley-Terry maximum-likelihood.

Prerequisites:
  - cutechess-cli on PATH or specified via --cutechess
  - uci_config_bridge.py integrated into uci.py (so engine accepts setoption)
  - An opening book (.epd, .pgn, or .bin)

Usage:
    python3 tune.py PARAM_NAME val1 val2 val3 ... [options]

Examples:
    python3 tune.py QS_SOFT_STOP_DIVISOR 7.0 8.0 9.0 10.0 --games 30
    python3 tune.py FUTILITY_MAX_DEPTH 2 3 4 --games 50 --tc 40/60+0.5
    python3 tune.py MAX_QS_MOVES "[12,6,4,2]" "[10,5,3,2]" "[15,8,5,3]"
    python3 tune.py ASPIRATION_WINDOW 50 75 100 --book openings.epd
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config

SCRIPT_DIR = Path(__file__).resolve().parent

# Ã¢â€¢â€Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢â€”
# Ã¢â€¢â€˜  Bradley-Terry Maximum-Likelihood Elo Estimation            Ã¢â€¢â€˜
# Ã¢â€¢Å¡Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â

class BradleyTerry:
    """
    Estimates Elo ratings from pairwise game results using the
    Bradley-Terry model solved by the MM (minorization-maximization)
    algorithm.

    Draws are handled by splitting: each draw counts as half a win
    for both sides.
    """

    def __init__(self, players: List[str]):
        self.players = list(players)
        self.n = len(self.players)
        self.idx = {p: i for i, p in enumerate(self.players)}
        # effective wins (draws count 0.5 each side)
        self.wins = [[0.0] * self.n for _ in range(self.n)]
        # total games between each pair
        self.games = [[0] * self.n for _ in range(self.n)]
        # raw W-L-D for display (from first player's perspective)
        self.raw: Dict[Tuple[int, int], Tuple[int, int, int]] = {}

    def add_result(self, player_a: str, player_b: str,
                   wins_a: int, wins_b: int, draws: int):
        """Record a match result (from player_a's perspective)."""
        i, j = self.idx[player_a], self.idx[player_b]
        self.wins[i][j] += wins_a + 0.5 * draws
        self.wins[j][i] += wins_b + 0.5 * draws
        total = wins_a + wins_b + draws
        self.games[i][j] += total
        self.games[j][i] += total
        # accumulate raw counts
        pw, pl, pd = self.raw.get((i, j), (0, 0, 0))
        self.raw[(i, j)] = (pw + wins_a, pl + wins_b, pd + draws)
        pw2, pl2, pd2 = self.raw.get((j, i), (0, 0, 0))
        self.raw[(j, i)] = (pw2 + wins_b, pl2 + wins_a, pd2 + draws)

    # Ã¢â€â‚¬Ã¢â€â‚¬ MLE Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

    def estimate(self, max_iter: int = 10_000, tol: float = 1e-8) \
            -> Dict[str, Tuple[float, float, float]]:
        """
        Returns {player_name: (elo, elo_95_lo, elo_95_hi)}.
        Elo is anchored so the mean across all players is 0.
        """
        n = self.n
        if n < 2:
            return {self.players[0]: (0.0, 0.0, 0.0)}

        gamma = [1.0] * n

        for _ in range(max_iter):
            prev = gamma[:]
            for i in range(n):
                W_i = sum(self.wins[i])
                if W_i == 0:
                    gamma[i] = 1e-10
                    continue
                denom = sum(
                    self.games[i][j] / (gamma[i] + gamma[j])
                    for j in range(n)
                    if j != i and self.games[i][j] > 0
                )
                gamma[i] = W_i / denom if denom > 0 else 1e-10

            # anchor: geometric mean = 1
            geo = math.exp(sum(math.log(g) for g in gamma) / n)
            gamma = [g / geo for g in gamma]

            if max(abs(gamma[k] - prev[k]) for k in range(n)) < tol:
                break

        elos = [400.0 * math.log10(max(g, 1e-15)) for g in gamma]
        se = self._standard_errors(gamma)

        return {
            self.players[i]: (
                elos[i],
                elos[i] - 1.96 * se[i],
                elos[i] + 1.96 * se[i],
            )
            for i in range(n)
        }

    def _standard_errors(self, gamma: List[float]) -> List[float]:
        """SE in Elo units via observed Fisher information."""
        n = self.n
        se_elo = []
        for i in range(n):
            fisher = 0.0
            for j in range(n):
                if i == j or self.games[i][j] == 0:
                    continue
                fisher += (self.games[i][j] * gamma[i] * gamma[j]
                           / (gamma[i] + gamma[j]) ** 2)
            if fisher > 0:
                se_log = 1.0 / math.sqrt(fisher)
                se_elo.append(se_log * 400.0 / math.log(10)
                              / max(gamma[i], 1e-15))
            else:
                se_elo.append(float('inf'))
        return se_elo

    # Ã¢â€â‚¬Ã¢â€â‚¬ display helpers Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

    def crosstable_str(self) -> str:
        """Formatted cross-table with W-L-D per pairing."""
        names = [p[:22] for p in self.players]
        cw = max(len(nm) for nm in names) + 2

        lines = []
        hdr = " " * cw + "".join(f"{nm:>14}" for nm in names)
        lines.append(hdr)
        lines.append("-" * len(hdr))

        for i, ni in enumerate(names):
            row = f"{ni:>{cw}}"
            for j in range(self.n):
                if i == j:
                    row += f"{'---':>14}"
                elif (i, j) in self.raw:
                    w, l, d = self.raw[(i, j)]
                    row += f"{w:>3}W {l}L {d}D   "
                else:
                    row += f"{'':>14}"
            lines.append(row)

        return "\n".join(lines)

    def player_wld(self, name: str) -> Tuple[int, int, int]:
        """Aggregate W-L-D for a player across all opponents."""
        i = self.idx[name]
        tw = tl = td = 0
        for j in range(self.n):
            if j == i:
                continue
            w, l, d = self.raw.get((i, j), (0, 0, 0))
            tw += w; tl += l; td += d
        return tw, tl, td


# Ã¢â€¢â€Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢â€”
# Ã¢â€¢â€˜  Likelihood of Superiority (pairwise)                       Ã¢â€¢â€˜
# Ã¢â€¢Å¡Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â

def compute_los(wins: float, losses: float) -> float:
    """
    Probability that true strength of A > B,
    using a normal approximation to the binomial.
    """
    total = wins + losses
    if total == 0:
        return 50.0
    p = wins / total
    se = math.sqrt(p * (1 - p) / total) if 0 < p < 1 else 0.001
    z = (p - 0.5) / se
    return 50.0 * (1.0 + math.erf(z / math.sqrt(2.0)))


# Ã¢â€¢â€Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢â€”
# Ã¢â€¢â€˜  cutechess-cli output parsing                               Ã¢â€¢â€˜
# Ã¢â€¢Å¡Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â

# Matches: "Finished game 1 (nf_INT8 vs nf_INT16): 1-0 {White mates}"
_FINISHED_RE = re.compile(
    r'Finished game \d+ \((\S+) vs (\S+)\):\s*(\S+)'
)

# Also try the "Score of X vs Y" lines (non-tournament or 2-engine mode)
_SCORE_RE = re.compile(
    r'Score of (\S+) vs (\S+):\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)'
)


def parse_cutechess_output(output: str,
                           player_names: List[str]) -> Optional[BradleyTerry]:
    """
    Parse cutechess-cli stdout and build a BradleyTerry model.

    Handles two output formats:
    1. Round-robin tournaments: "Finished game N (A vs B): result {reason}"
    2. Head-to-head matches:    "Score of A vs B: W - L - D [pct] N"
    """
    name_set = set(player_names)

    # Ã¢â€â‚¬Ã¢â€â‚¬ collect pairwise W-L-D from individual game results Ã¢â€â‚¬Ã¢â€â‚¬
    pair_results: Dict[Tuple[str, str], List[int]] = {}  # (a,b) -> [wins_a, wins_b, draws]

    for line in output.splitlines():
        # Try "Finished game" lines first (round-robin)
        m = _FINISHED_RE.search(line)
        if m:
            pa, pb, result = m.group(1), m.group(2), m.group(3)
            if pa not in name_set or pb not in name_set:
                continue

            key = (pa, pb)
            if key not in pair_results:
                pair_results[key] = [0, 0, 0]

            if result == '1-0':
                pair_results[key][0] += 1      # pa wins
            elif result == '0-1':
                pair_results[key][1] += 1      # pb wins
            elif result in ('1/2-1/2', '*'):
                pair_results[key][2] += 1      # draw
            continue

        # Fall back to "Score of" summary lines (2-engine mode)
        m = _SCORE_RE.search(line)
        if m:
            pa, pb = m.group(1), m.group(2)
            w, l, d = int(m.group(3)), int(m.group(4)), int(m.group(5))
            if pa in name_set and pb in name_set:
                key = (pa, pb)
                if key not in pair_results:
                    pair_results[key] = [0, 0, 0]
                pair_results[key][0] += w
                pair_results[key][1] += l
                pair_results[key][2] += d

    if not pair_results:
        return None

    bt = BradleyTerry(player_names)
    for (pa, pb), (w, l, d) in pair_results.items():
        bt.add_result(pa, pb, w, l, d)

    return bt


# Ã¢â€¢â€Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢â€”
# Ã¢â€¢â€˜  Polyglot .bin Ã¢â€ â€™ EPD converter                              Ã¢â€¢â€˜
# Ã¢â€¢Å¡Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â

def polyglot_to_epd(bin_path: str, output_path: str,
                    num_positions: int = 200, max_ply: int = 12) -> int:
    """
    Sample opening positions from a Polyglot book by playing out
    weighted-random book lines.  Returns number of positions written.
    Requires python-chess.
    """
    import random
    try:
        import chess
        import chess.polyglot
    except ImportError:
        print("Error: python-chess is required for Polyglot conversion.")
        print("       pip install chess")
        sys.exit(1)

    positions = set()
    max_attempts = num_positions * 30

    with chess.polyglot.open_reader(bin_path) as reader:
        for _ in range(max_attempts):
            if len(positions) >= num_positions:
                break
            board = chess.Board()
            for _ in range(max_ply):
                entries = list(reader.find_all(board))
                if not entries:
                    break
                total_w = sum(e.weight for e in entries)
                if total_w == 0:
                    break
                r = random.randint(0, total_w - 1)
                cum = 0
                chosen = entries[0]
                for e in entries:
                    cum += e.weight
                    if cum > r:
                        chosen = e
                        break
                board.push(chosen.move)
            if board.fullmove_number >= 4:
                positions.add(board.epd())

    with open(output_path, 'w') as f:
        for epd in sorted(positions):
            f.write(epd + "\n")
    return len(positions)


# Ã¢â€¢â€Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢â€”
# Ã¢â€¢â€˜  cutechess-cli command builder                              Ã¢â€¢â€˜
# Ã¢â€¢Å¡Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â

def build_cutechess_cmd(args, player_names: List[str],
                        param_name: str, values: List[str]) -> List[str]:
    """Assemble the full cutechess-cli invocation."""

    cmd: List[str] = [args.cutechess, '-tournament', 'round-robin']

    # â”€â”€ per-engine configuration â”€â”€
    for name, val in zip(player_names, values):
        engine_args = ['-engine',
                f'cmd={args.engine_cmd}',
                f'name={name}',
                f'option.{param_name}={val}']
        if args.ponder:
            engine_args.append('ponder')
        cmd += engine_args

    # Ã¢â€â‚¬Ã¢â€â‚¬ common engine settings Ã¢â€â‚¬Ã¢â€â‚¬
    each_args = ['proto=uci', f'tc={args.tc}', f'timemargin={args.timemargin}']
    # Don't add Threads to -each if we're tuning Threads (per-engine values take precedence)
    if args.threads is not None and param_name.upper() != 'THREADS':
        each_args.append(f'option.Threads={args.threads}')
    cmd += ['-each'] + each_args

    # Ã¢â€â‚¬Ã¢â€â‚¬ opening book Ã¢â€â‚¬Ã¢â€â‚¬
    book_path = args.book
    if book_path:
        if book_path.endswith('.bin'):
            epd_path = tempfile.mktemp(suffix='.epd', prefix='tune_book_')
            print(f"Converting Polyglot book Ã¢â€ â€™ EPD ...")
            n = polyglot_to_epd(book_path, epd_path,
                                num_positions=max(200, args.games * 3))
            print(f"  Extracted {n} positions Ã¢â€ â€™ {epd_path}")
            book_path = epd_path

        fmt = 'epd' if book_path.endswith('.epd') else 'pgn'
        cmd += ['-openings', f'file={book_path}', f'format={fmt}',
                'policy=round']

    # Ã¢â€â‚¬Ã¢â€â‚¬ rounds & pairing Ã¢â€â‚¬Ã¢â€â‚¬
    rounds = max(1, args.games // 2)
    cmd += ['-games', '2',             # each opening from both sides
            '-rounds', str(rounds),
            '-repeat']                  # same opening for the colour-swap

    # Ã¢â€â‚¬Ã¢â€â‚¬ adjudication Ã¢â€â‚¬Ã¢â€â‚¬
    cmd += ['-draw',
            f'movenumber={args.draw_movenumber}',
            f'movecount={args.draw_movecount}',
            f'score={args.draw_score}']
    cmd += ['-resign',
            f'movecount={args.resign_movecount}',
            f'score={args.resign_score}']
    cmd += ['-maxmoves', str(args.maxmoves)]
    cmd += ['-recover']

    # Ã¢â€â‚¬Ã¢â€â‚¬ concurrency (sequential) Ã¢â€â‚¬Ã¢â€â‚¬
    cmd += ['-concurrency', '1']

    # Ã¢â€â‚¬Ã¢â€â‚¬ PGN output Ã¢â€â‚¬Ã¢â€â‚¬
    if args.pgnout:
        cmd += ['-pgnout', args.pgnout]

    return cmd


# Ã¢â€¢â€Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢â€”
# Ã¢â€¢â€˜  Result display                                             Ã¢â€¢â€˜
# Ã¢â€¢Å¡Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â

def print_results(param_name: str, values: List[str],
                  player_names: List[str], bt: BradleyTerry,
                  elo_results: Dict[str, Tuple[float, float, float]]):

    # Ã¢â€â‚¬Ã¢â€â‚¬ cross-table Ã¢â€â‚¬Ã¢â€â‚¬
    print(f"\n{'=' * 80}")
    print(f"  CROSS TABLE")
    print(f"{'=' * 80}\n")
    print(bt.crosstable_str())

    # Ã¢â€â‚¬Ã¢â€â‚¬ ranked Elo table Ã¢â€â‚¬Ã¢â€â‚¬
    print(f"\n{'=' * 80}")
    print(f"  TUNING RESULTS Ã¢â‚¬â€ {param_name}")
    print(f"{'=' * 80}\n")

    rows = []
    for name, val in zip(player_names, values):
        elo, lo, hi = elo_results[name]
        w, l, d = bt.player_wld(name)
        games = w + l + d
        score_pct = (w + 0.5 * d) / games * 100 if games else 50.0
        los_pct = compute_los(w, l)
        rows.append((val, elo, lo, hi, w, l, d, games, score_pct, los_pct))

    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"  {'Value':<25} {'Elo':>7} {'Ã‚Â± 95%':>8}"
          f"  {'W-L-D':>12} {'Score%':>7} {'LOS%':>6}")
    print(f"  {'-' * 72}")

    for i, (val, elo, lo, hi, w, l, d, g, sp, los) in enumerate(rows):
        err = (hi - lo) / 2
        wld = f"{w}-{l}-{d}"
        marker = "  <<<" if i == 0 else ""
        print(f"  {val:<25} {elo:>+7.1f} {err:>7.1f}"
              f"  {wld:>12} {sp:>6.1f}% {los:>5.1f}%{marker}")

    print(f"  {'-' * 72}")

    # Ã¢â€â‚¬Ã¢â€â‚¬ best value summary Ã¢â€â‚¬Ã¢â€â‚¬
    best = rows[0]
    print(f"\n  BEST: {param_name} = {best[0]}")
    print(f"        Elo: {best[1]:+.1f}  "
          f"95% CI: [{best[2]:+.1f}, {best[3]:+.1f}]")
    print(f"        W-L-D: {best[4]}-{best[5]}-{best[6]}  "
          f"Score: {best[8]:.1f}%  LOS: {best[9]:.1f}%")

    if len(rows) > 1:
        gap = best[1] - rows[1][1]
        print(f"        Gap to 2nd ({rows[1][0]}): {gap:+.1f} Elo")

    # Ã¢â€â‚¬Ã¢â€â‚¬ significance warning Ã¢â€â‚¬Ã¢â€â‚¬
    if len(rows) > 1:
        top_lo = best[2]
        second_hi = rows[1][3]
        if top_lo < second_hi:
            print(f"\n  Ã¢Å¡Â   Confidence intervals of top-2 overlap Ã¢â‚¬â€ "
                  f"consider more games for significance.")

    print(f"\n{'=' * 80}\n")


# Ã¢â€¢â€Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢â€”
# Ã¢â€¢â€˜  Main                                                       Ã¢â€¢â€˜
# Ã¢â€¢Å¡Ã¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢ÂÃ¢â€¢Â

def main():
    parser = argparse.ArgumentParser(
        description="Self-play parameter tuning via round-robin tournament.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s QS_SOFT_STOP_DIVISOR 7.0 8.0 9.0 10.0
  %(prog)s FUTILITY_MAX_DEPTH 2 3 4 --games 50
  %(prog)s MAX_QS_MOVES "[12,6,4,2]" "[10,5,3,2]" "[15,8,5,3]"
  %(prog)s ASPIRATION_WINDOW 50 75 100 --book openings.epd --tc 40/60+0.5
        """,
    )

    parser.add_argument("param_name",
                        help="UCI parameter name to tune")
    parser.add_argument("values", nargs="+",
                        help='Values to test (quote lists: "[12,6,4,2]")')

    # tournament
    grp = parser.add_argument_group("tournament settings")
    grp.add_argument("--games", "-g", type=int, default=30,
                     help="Games per pair (default: 30)")
    grp.add_argument("--tc", default="40/120+1",
                     help="Time control (default: 40/120+1)")
    grp.add_argument("--threads", "-t", type=int, default=config.THREADS,
                     help="UCI Threads per engine instance")
    grp.add_argument("--book", "-b", default=f"{SCRIPT_DIR}/../book/komodo.bin",
                     help="Opening book (.epd, .pgn, or .bin)")
    grp.add_argument("--ponder", dest="ponder",
                     default=config.PONDERING_ENABLED,
                     action=argparse.BooleanOptionalAction,
                     help=f"Enable pondering (default: {config.PONDERING_ENABLED})")

    # paths
    grp2 = parser.add_argument_group("paths")
    grp2.add_argument("--engine-cmd", "-e", default=None,
                      help="Path to engine executable (default: auto-detect uci.sh)")
    grp2.add_argument("--cutechess", "-c", default=None,
                      help="Path to cutechess-cli (default: auto-detect)")

    # adjudication
    grp3 = parser.add_argument_group("adjudication")
    grp3.add_argument("--draw-movenumber", type=int, default=40)
    grp3.add_argument("--draw-movecount", type=int, default=5)
    grp3.add_argument("--draw-score", type=int, default=50)
    grp3.add_argument("--resign-movecount", type=int, default=3)
    grp3.add_argument("--resign-score", type=int, default=500)
    grp3.add_argument("--maxmoves", type=int, default=100)
    grp3.add_argument("--timemargin", type=int, default=9999)

    # output
    grp4 = parser.add_argument_group("output")
    grp4.add_argument("--pgnout", default=None,
                      help="PGN output file (default: auto tmp)")

    args = parser.parse_args()

    # Ã¢â€â‚¬Ã¢â€â‚¬ auto-detect engine Ã¢â€â‚¬Ã¢â€â‚¬
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not args.engine_cmd:
        for cand in [
            os.path.join(script_dir, "uci.sh"),
            os.path.join(script_dir, "..", "uci.sh"),
            os.path.join(script_dir, "..", "src", "uci.sh"),
        ]:
            if os.path.isfile(cand):
                args.engine_cmd = os.path.abspath(cand)
                break
        if not args.engine_cmd:
            print("Error: Could not find uci.sh Ã¢â‚¬â€ use --engine-cmd")
            sys.exit(1)

    # Ã¢â€â‚¬Ã¢â€â‚¬ auto-detect cutechess-cli Ã¢â€â‚¬Ã¢â€â‚¬
    if not args.cutechess:
        if shutil.which("cutechess-cli"):
            args.cutechess = "cutechess-cli"
        else:
            for cand in [
                os.path.join(script_dir, "..", "..", "cutechess",
                             "build", "cutechess-cli"),
                os.path.join(script_dir, "..", "cutechess-cli"),
            ]:
                if os.path.isfile(cand):
                    args.cutechess = os.path.abspath(cand)
                    break
        if not args.cutechess:
            print("Error: Could not find cutechess-cli Ã¢â‚¬â€ use --cutechess")
            sys.exit(1)

    if not args.pgnout:
        args.pgnout = tempfile.mktemp(suffix='.pgn', prefix='tune_')

    # Ã¢â€â‚¬Ã¢â€â‚¬ player names (must be unique for cutechess) Ã¢â€â‚¬Ã¢â€â‚¬
    player_names = [f"nf_{v}" for v in args.values]
    if len(set(player_names)) != len(player_names):
        player_names = [f"nf_{i}_{v}" for i, v in enumerate(args.values)]

    # Ã¢â€â‚¬Ã¢â€â‚¬ plan summary Ã¢â€â‚¬Ã¢â€â‚¬
    n = len(args.values)
    pairs = n * (n - 1) // 2
    total_games = pairs * args.games

    print(f"\n{'#' * 70}")
    print(f"#  SELF-PLAY PARAMETER TUNING")
    print(f"#  Parameter:     {args.param_name}")
    print(f"#  Values:        {args.values}")
    print(f"#  Engines:       {n}")
    print(f"#  Pairings:      {pairs}")
    print(f"#  Games/pair:    {args.games}")
    print(f"#  Total games:   {total_games}")
    print(f"#  Time control:  {args.tc}")
    print(f"#  Engine:        {args.engine_cmd}")
    print(f"#  cutechess-cli: {args.cutechess}")
    print(f"#  Opening book:  {args.book or '(none)'}")
    print(f"#  Ponder:        {args.ponder}")
    print(f"#  PGN output:    {args.pgnout}")
    print(f"{'#' * 70}\n")

    # Ã¢â€â‚¬Ã¢â€â‚¬ build & display command Ã¢â€â‚¬Ã¢â€â‚¬
    cmd = build_cutechess_cmd(args, player_names,
                              args.param_name, args.values)

    print("Command:")
    # pretty-print with line continuations
    cmd_str_parts = []
    for c in cmd:
        cmd_str_parts.append(c)
    print("  " + " \\\n    ".join(cmd_str_parts))
    print()

    # ── run cutechess-cli ──
    output_lines: List[str] = []

    # Progress tracking for 3+ player tournaments
    player_stats: Dict[str, List[int]] = {name: [0, 0, 0] for name in player_names}  # [W, L, D]
    games_completed = 0
    progress_interval = max(1, total_games // 20)  # Show progress ~20 times, minimum every game

    def update_progress(line: str) -> bool:
        """Parse finished game line and update stats. Returns True if updated."""
        nonlocal games_completed
        m = _FINISHED_RE.search(line)
        if not m:
            return False

        pa, pb, result = m.group(1), m.group(2), m.group(3)
        if pa not in player_stats or pb not in player_stats:
            return False

        games_completed += 1

        if result == '1-0':
            player_stats[pa][0] += 1  # pa wins
            player_stats[pb][1] += 1  # pb loses
        elif result == '0-1':
            player_stats[pa][1] += 1  # pa loses
            player_stats[pb][0] += 1  # pb wins
        elif result in ('1/2-1/2', '*'):
            player_stats[pa][2] += 1  # draw
            player_stats[pb][2] += 1
        return True

    def print_progress():
        """Print current standings."""
        # Sort by score (wins + 0.5*draws)
        ranked = sorted(
            player_stats.items(),
            key=lambda x: x[1][0] + 0.5 * x[1][2],
            reverse=True
        )
        parts = []
        for name, (w, l, d) in ranked:
            score = w + 0.5 * d
            total = w + l + d
            pct = (score / total * 100) if total > 0 else 0
            parts.append(f"{name}: +{w}-{l}={d} ({pct:.0f}%)")
        print(f"  ── Progress {games_completed}/{total_games}: {' | '.join(parts)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)

            # Track progress for 3+ player tournaments
            if n >= 3 and update_progress(line):
                if games_completed % progress_interval == 0 or games_completed == total_games:
                    print_progress()

        process.wait()
        output = ''.join(output_lines)

        if process.returncode != 0:
            print(f"\nWarning: cutechess-cli exited with code "
                  f"{process.returncode}")

    except FileNotFoundError:
        print(f"Error: Could not execute: {cmd[0]}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted Ã¢â‚¬â€ parsing partial results Ã¢â‚¬Â¦")
        process.terminate()
        output = ''.join(output_lines)

    # Ã¢â€â‚¬Ã¢â€â‚¬ check for engine failures Ã¢â€â‚¬Ã¢â€â‚¬
    if "Could not initialize player" in output:
        print("\n*** ENGINE INITIALIZATION FAILED ***")
        print("Verify that uci.sh starts correctly and that")
        print("uci_config_bridge.py is integrated (the engine must")
        print(f"accept  setoption name {args.param_name} value Ã¢â‚¬Â¦).")
        sys.exit(1)

    # Ã¢â€â‚¬Ã¢â€â‚¬ parse & analyse Ã¢â€â‚¬Ã¢â€â‚¬
    bt = parse_cutechess_output(output, player_names)

    if not bt:
        print("\nError: Could not parse any results from cutechess output.")
        print("Check that cutechess-cli ran correctly.")
        sys.exit(1)

    elo_results = bt.estimate()
    print_results(args.param_name, args.values, player_names, bt, elo_results)

    print(f"PGN saved to: {args.pgnout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())