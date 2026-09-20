"""Layered index behaviour tests (isolated rag-service subprocess, CI-safe)."""

from conftest import run_service_case


def test_rag_layered_behavioural_checks_pass():
    r = run_service_case("rag_layers_case")
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert "RAG_LAYER_CHECKS passed=13 failed=0" in r.stdout