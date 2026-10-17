import pytest

from scripts.deploy_procs import per_process_cap


def test_per_process_cap_is_an_integer_floor_of_at_least_one():
    assert per_process_cap(32, 4) == 8
    assert per_process_cap(32, 1) == 32
    assert per_process_cap(3, 4) == 1          # 3 // 4 == 0, but a process must be able to take a call
    assert per_process_cap(0, 2) == 1
    assert per_process_cap(100, 6) == 16


def test_per_process_cap_rejects_fewer_than_one_process():
    with pytest.raises(ValueError):
        per_process_cap(32, 0)
    with pytest.raises(ValueError):
        per_process_cap(32, -3)
