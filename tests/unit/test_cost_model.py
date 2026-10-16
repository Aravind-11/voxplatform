import math

from scripts import cost_model as cm


def test_gpus_are_sized_for_peak_plus_spare_with_a_floor_of_two():
    assert cm.gpus_needed(5, 24) == 2 and cm.gpus_needed(24, 24) == 2 and cm.gpus_needed(25, 24) == 3
    assert cm.gpus_needed(100, 24) == 6 and cm.gpus_needed(100, 24, spare=0) == 5


def test_cost_per_minute_is_gpu_spend_over_minutes_spoken():
    s = cm.scenario(24, 12, gpu_hourly=2.0, calls_per_gpu=24, telephony=0.01)
    assert s["gpus"] == 2
    assert math.isclose(s["monthly_gpu"], 2 * 2.0 * 730)
    assert math.isclose(s["minutes"], 12 * 60 * 730)
    assert math.isclose(s["gpu_per_min"], (2 * 2.0 * 730) / (12 * 60 * 730))     # = 2*2/(12*60) = 0.005556
    assert math.isclose(s["total_per_min"], s["gpu_per_min"] + 0.01)


def test_break_even_matches_a_hand_calculation():
    # one $3.60/h GPU = $0.06/min; paid 0.05, phone 0.01 -> 0.04 headroom per call-minute -> 1.5 calls
    assert math.isclose(cm.break_even_calls(3.60, 0.01, 0.05), 1.5)
    assert cm.break_even_calls(3.60, 0.06, 0.05) == float("inf")                  # phone line alone exceeds the paid price


def test_render_has_a_row_per_scenario():
    t = cm.render(2.86, 24, 0.0085, 0.05)
    assert t.count("\n| ") >= 18 and "cheaper" in t
    assert "dearer" in cm.render(20.0, 24, 0.0085, 0.05)                 # an expensive GPU flips small scenarios
