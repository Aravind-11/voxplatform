import pytest

from scripts.deploy_procs import per_process_cap


def test_per_process_cap_is_an_integer_floor():
    assert per_process_cap(32, 4) == 8
    assert per_process_cap(32, 1) == 32
    assert per_process_cap(100, 6) == 16
    assert 4 * per_process_cap(32, 4) <= 32
    assert 6 * per_process_cap(100, 6) <= 100


def test_per_process_cap_rejects_fewer_than_one_process():
    with pytest.raises(ValueError):
        per_process_cap(32, 0)
    with pytest.raises(ValueError):
        per_process_cap(32, -3)


def test_per_process_cap_sum_does_not_exceed_the_node_cap():
    # Rounding a 0 quotient up to 1 lets procs processes admit more than node_max.
    with pytest.raises(ValueError):
        per_process_cap(3, 4)
    with pytest.raises(ValueError):
        per_process_cap(1, 2)
    # A zero node cap stays 0. Bumping each process to 1 would admit procs calls.
    assert per_process_cap(0, 2) == 0
    assert per_process_cap(0, 1) == 0
    assert per_process_cap(5, 3) == 1
    assert 3 * per_process_cap(5, 3) <= 5
    with pytest.raises(ValueError):
        per_process_cap(-4, 2)
