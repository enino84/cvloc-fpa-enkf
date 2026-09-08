# -*- coding: utf-8 -*-
"""
Progress reporting designed to be followed from a remote machine with
``docker logs -f`` or ``tail -f``.

Rules this module follows, because logs read over a slow link are useless
otherwise:

* every line is timestamped and flushed immediately, never buffered;
* every line carries the experiment id, so interleaved output from parallel
  runs is still attributable;
* long loops report a running count, the elapsed time and an estimate of the
  time remaining, so a run that is going to take four hours announces it in
  the first minute rather than at the end;
* a heartbeat is printed even when nothing has finished yet, so a silent
  process can be told apart from a hung one.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import timedelta

_T0 = time.time()


def _stamp() -> str:
    now = time.strftime("%H:%M:%S")
    el = timedelta(seconds=int(time.time() - _T0))
    return f"{now} (+{el})"


def log(msg: str, exp_id: str = "", level: str = "INFO") -> None:
    tag = f"[{exp_id}] " if exp_id else ""
    sys.stdout.write(f"{_stamp()} {level:<5s} {tag}{msg}\n")
    sys.stdout.flush()


def banner(title: str, lines=None, exp_id: str = "") -> None:
    width = 78
    sys.stdout.write("\n" + "=" * width + "\n")
    sys.stdout.write(f"  {title}\n")
    if lines:
        for ln in lines:
            sys.stdout.write(f"  {ln}\n")
    sys.stdout.write("=" * width + "\n")
    sys.stdout.flush()


def fmt_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "unknown"
    return str(timedelta(seconds=int(seconds)))


class Progress:
    """Counter with elapsed time and an estimate of the time remaining.

    Usage::

        p = Progress(total=120, label="cells", exp_id="EXP-02-META")
        for ...:
            ...
            p.step("sa scen=0 run=1 rmse=0.0031")
        p.done()
    """

    def __init__(self, total: int, label: str = "steps", exp_id: str = "",
                 every: int = 1, heartbeat_s: float = 60.0):
        self.total = int(total)
        self.label = label
        self.exp_id = exp_id
        self.every = max(1, int(every))
        self.heartbeat_s = heartbeat_s
        self.n = 0
        self.t0 = time.time()
        self.last_emit = self.t0
        log(f"starting {self.total} {label}", exp_id)

    def step(self, detail: str = "", force: bool = False) -> None:
        self.n += 1
        now = time.time()
        due = (self.n % self.every == 0
               or self.n == self.total
               or force
               or now - self.last_emit >= self.heartbeat_s)
        if not due:
            return
        el = now - self.t0
        rate = self.n / el if el > 0 else 0.0
        remaining = (self.total - self.n) / rate if rate > 0 else float("nan")
        pct = 100.0 * self.n / self.total if self.total else 100.0
        bar_n = int(pct // 5)
        bar = "#" * bar_n + "." * (20 - bar_n)
        msg = (f"[{bar}] {pct:5.1f}%  {self.n}/{self.total} {self.label}  "
               f"elapsed {fmt_eta(el)}  eta {fmt_eta(remaining)}")
        if detail:
            msg += f"  | {detail}"
        log(msg, self.exp_id)
        self.last_emit = now

    def note(self, message):
        """Log a line without advancing the counter.

        Used for work that happens inside a cell but is not a cell of its own,
        such as the oracle sweep EXP-09 runs once per regime.
        """
        log(message, self.exp_id)

    def done(self) -> float:
        el = time.time() - self.t0
        log(f"finished {self.n}/{self.total} {self.label} in {fmt_eta(el)}",
            self.exp_id)
        return el


def announce_plan(exp_id: str, scale_name: str, n_cells: int,
                  seconds_per_cell_guess: float = None) -> None:
    """Print the size of the job before it starts.

    A run that will take hours should say so in its first lines, not at the
    end. The estimate is deliberately labelled as a guess.
    """
    lines = [f"scale: {scale_name}", f"cells to run: {n_cells}"]
    if seconds_per_cell_guess:
        est = n_cells * seconds_per_cell_guess
        lines.append(f"rough estimate: {fmt_eta(est)} "
                     f"(at {seconds_per_cell_guess:.1f}s per cell)")
    banner(f"{exp_id}", lines)


def env_summary() -> None:
    keys = ["SCALE", "PYTHONHASHSEED", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RESULTS_DIR"]
    parts = [f"{k}={os.environ.get(k, 'unset')}" for k in keys]
    log("environment: " + "  ".join(parts))
