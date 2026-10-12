import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.limits import _linux_group_rss_kib, run_limited


def test_linux_group_memory(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.limits.os.sysconf", lambda name: 4096)
    # /proc stat fields 3..24; include a process name with spaces and ')'.
    for pid, group, pages in ((1, 100, 10), (2, 100, 20), (3, 200, 50)):
        directory = tmp_path / str(pid)
        directory.mkdir()
        fields = ["S", "1", str(group)] + ["0"] * 18 + [str(pages)]
        (directory / "stat").write_text(f"{pid} (worker ) name) " + " ".join(fields))
    (tmp_path / "4").mkdir()  # An exited process has no stat file.
    assert _linux_group_rss_kib(100, tmp_path) == 120


@pytest.mark.skipif(sys.platform != "linux", reason="Linux container regression")
def test_linux_monitor_without_ps(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    result = run_limited([sys.executable, "-c", "print('done')"], timeout_s=5, memory_mib=256)
    assert result.returncode == 0
    assert result.stdout == "done\n"
    assert result.limit is None


def test_limited_command_completes():
    result = run_limited([sys.executable, "-c", "print('done')"], timeout_s=5, memory_mib=256)
    assert result.returncode == 0
    assert result.stdout == "done\n"
    assert result.limit is None


def test_timeout_kills_children():
    code = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "print(p.pid,flush=True); time.sleep(30)"
    )
    result = run_limited([sys.executable, "-c", code], timeout_s=0.5, memory_mib=256)
    assert result.limit == "timeout"
    assert result.elapsed_s < 5
    pid = int(result.stdout)
    if sys.platform == "linux":
        stat = Path(f"/proc/{pid}/stat")
        try:
            assert stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        except FileNotFoundError:
            pass
    else:
        status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
        assert not status.stdout.strip() or status.stdout.strip().startswith("Z")


def test_memout_includes_children():
    child = "import time; data=bytearray(96*1024*1024); time.sleep(30)"
    code = f"import subprocess,sys; subprocess.run([sys.executable,'-c',{child!r}])"
    result = run_limited([sys.executable, "-c", code], timeout_s=10, memory_mib=80)
    assert result.limit == "memout"
    assert result.peak_rss_mib > 80
    assert result.elapsed_s < 5
