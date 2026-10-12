"""Wall-clock and aggregate RSS limits for an isolated process group (POSIX)."""

import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LimitedRun:
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    peak_rss_mib: float
    limit: str | None


def _linux_group_rss_kib(pgid, proc=Path("/proc")):
    page_kib = os.sysconf("SC_PAGE_SIZE") // 1024
    rss_kib = 0
    for stat in proc.glob("[0-9]*/stat"):
        try:
            # comm is parenthesized and may itself contain spaces or ')'.
            fields = stat.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid:
                rss_kib += int(fields[21]) * page_kib
        except (OSError, IndexError, ValueError):
            # Processes can exit while the snapshot is being collected.
            continue
    return rss_kib


def _group_rss_kib(pgid):
    if sys.platform == "linux":
        return _linux_group_rss_kib(pgid)
    snapshot = subprocess.run(
        ["ps", "-axo", "pgid=,rss="], capture_output=True, text=True, check=True, timeout=5
    )
    return sum(int(rss) for group, rss in map(str.split, snapshot.stdout.splitlines()) if int(group) == pgid)


def run_limited(command, *, timeout_s=120, memory_mib=4096, cwd=None):
    """Limit the whole process group; RSS is sampled every 100 ms."""
    if timeout_s <= 0 or memory_mib <= 0:
        raise ValueError("Time and memory limits must be positive")
    start = time.monotonic()
    peak_kib, limit = 0, None
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while process.poll() is None:
                remaining = timeout_s - (time.monotonic() - start)
                if remaining <= 0:
                    limit = "timeout"
                    break
                peak_kib = max(peak_kib, _group_rss_kib(process.pid))
                if peak_kib > memory_mib * 1024:
                    limit = "memout"
                    break
                try:
                    process.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # Children may still be alive even if the group leader exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        stdout.seek(0)
        stderr.seek(0)
        return LimitedRun(
            process.returncode,
            stdout.read().decode(errors="replace"),
            stderr.read().decode(errors="replace"),
            time.monotonic() - start,
            peak_kib / 1024,
            limit,
        )
