"""Grounded mock provider behavioural tests (isolated gateway subprocess)."""

from conftest import run_service_case


def test_mock_provider_behavioural_checks_pass():
    r = run_service_case("mock_case")
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert "MOCK_CHECKS" in r.stdout and "failed=0" in r.stdout