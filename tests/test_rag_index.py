"""RAG index behavioural tests (ran in an isolated rag-service subprocess)."""

from conftest import run_service_case


def test_rag_behavioural_checks_pass():
    r = run_service_case("rag_case")
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert "RAG_CHECKS passed=8 failed=0" in r.stdout