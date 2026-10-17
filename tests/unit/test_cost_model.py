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
    assert "$4,176" in t                                                 # 2 x $2.86 x 730h, unchanged by the default floor


def test_spot_above_the_warm_floor_and_overflow_minutes_match_a_hand_calculation():
    # peak 100 / 24 -> ceil 5 + 1 spare = 6. Floor 2 does not shrink that.
    # 2 on-demand at $2/h + 4 spot at $1/h = $8/h * 730 = $5840. All-on-demand would be 6*2*730 = $8760.
    s = cm.scenario(100, 50, gpu_hourly=2.0, calls_per_gpu=24, telephony=0.01, spot_hourly=1.0, warm_floor=2)
    assert (s["gpus"], s["on_demand"], s["spot"], s["warm_floor"]) == (6, 2, 4, 2)
    assert math.isclose(s["monthly_gpu"], (2 * 2.0 + 4 * 1.0) * 730)
    # A floor above the peak sizing is extra idle GPUs, all on-demand (nothing sits above the floor).
    small = cm.scenario(10, 4, gpu_hourly=2.0, calls_per_gpu=24, telephony=0.01, spot_hourly=1.0, warm_floor=4)
    assert (small["gpus"], small["on_demand"], small["spot"]) == (4, 4, 0)
    assert math.isclose(small["monthly_gpu"], 4 * 2.0 * 730)
    # peak 100, floor holds 2*24=48, excess 52 calls. Load factor avg/peak = 0.5.
    # 52 * 0.5 * 60 * 730 = 1,138,800 minutes, priced at the paid rate by the caller.
    assert math.isclose(cm.overflow_minutes(100, 24, 2, 50), 52 * 0.5 * 60 * 730)
    assert cm.overflow_minutes(40, 24, 2, 20) == 0                       # peak fits on the floor
    plain = cm.scenario(24, 12, gpu_hourly=2.0, calls_per_gpu=24, telephony=0.01)
    assert plain["gpus"] == 2 and math.isclose(plain["monthly_gpu"], 2 * 2.0 * 730)
    text = cm.render(2.86, 24, 0.0085, 0.05, spot_hourly=1.0, warm_floor=2)
    assert "Warm floor 2" in text and "spot nodes" in text and "overflow minutes" in text and "0.050/min" in text
