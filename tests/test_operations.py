from conftest import run_service_case

def test_operations_control_plane():
    result=run_service_case('operations')
    assert result.returncode==0,result.stdout+result.stderr

def test_operations_gateway_resilience():
    result=run_service_case('operations_gateway')
    assert result.returncode==0,result.stdout+result.stderr
