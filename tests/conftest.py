import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "tests" / "_cases"

# evaluation's `framework` package does not collide with any service's `app` package
sys.path.insert(0, str(ROOT / "services" / "evaluation"))


def run_service_case(case: str) -> subprocess.CompletedProcess:
    """Run a behaviour case in an isolated subprocess (each service owns its own `app` package)."""
    exe = ROOT / ".venv" / "bin" / "python" if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable
    return subprocess.run(
        [str(exe), str(CASES / f"{case}.py")],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
    )