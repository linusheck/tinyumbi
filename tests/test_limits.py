import subprocess
import sys

from evaluation.limits import run_limited


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
    status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    assert not status.stdout.strip() or status.stdout.strip().startswith("Z")


def test_memout_includes_children():
    child = "import time; data=bytearray(96*1024*1024); time.sleep(30)"
    code = f"import subprocess,sys; subprocess.run([sys.executable,'-c',{child!r}])"
    result = run_limited([sys.executable, "-c", code], timeout_s=10, memory_mib=80)
    assert result.limit == "memout"
    assert result.peak_rss_mib > 80
    assert result.elapsed_s < 5
