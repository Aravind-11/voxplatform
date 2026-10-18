"""Cost per call-minute from your own inputs. Prints markdown tables; nothing here is measured, it is arithmetic on inputs.

    python scripts/cost_model.py                       # defaults below
    python scripts/cost_model.py --gpu-hourly 3.78 --calls-per-gpu 24 --telephony 0.0085 --paid 0.05

Method: GPUs are sized for the PEAK number of simultaneous calls (peak / calls-per-gpu, rounded up) plus spare nodes for
failover, and you pay for them around the clock. Minutes actually spoken come from the AVERAGE number of simultaneous
calls. So cost per call-minute = (GPUs x $/hour x hours) / (average calls x 60 x hours). Idle capacity is the cost.
A warm floor (--warm-floor, default 2) is on-demand GPUs left running because a cold GPU takes minutes to load the
model. Billed GPUs are max(that sizing, the floor). With --spot-hourly, nodes above the floor are priced at spot and
the floor stays on-demand. Minutes the floor cannot absorb are printed at the paid rate. This prices a floor you
choose to keep warm; it does not remove idle-GPU cost or cold-start time.
Defaults: Lambda H100 PCIe $2.86/h (third-party price page, Feb 2026), Twilio US inbound $0.0085/min (third-party
summary, Aug 2026; check Twilio's own page), paid Ultravox $0.05/min (from the task brief; unknown what it includes),
24 calls per GPU (comfortable limit measured with synthetic callers on 1x H100: docs/FINDINGS.md).
Not included: engineering and on-call time, the database/proxy host, storage, bandwidth, Twilio Media Streams fees,
and the cost of any quality gap.
"""
from __future__ import annotations

import argparse
import math

HOURS_PER_MONTH = 730.0


def gpus_needed(peak_calls: float, calls_per_gpu: float, spare: int = 1, minimum: int = 2) -> int:
    """GPU nodes for a peak, with `spare` extra for failover/rolling updates and never fewer than `minimum` (a single
    node is a single point of failure, and a new node needs minutes to load the model)."""
    return max(minimum, math.ceil(peak_calls / calls_per_gpu) + spare)


def scenario(peak: float, avg: float, *, gpu_hourly: float, calls_per_gpu: float, telephony: float, spare: int = 1,
             spot_hourly: float | None = None, warm_floor: int | None = None) -> dict:
    """Bill max(peak sizing, warm floor). Spot, when set, prices only the nodes above the floor."""
    floor = max(0, int(warm_floor or 0))
    n = max(gpus_needed(peak, calls_per_gpu, spare), floor)
    if spot_hourly is None:
        on_demand, spot_n = n, 0
        monthly_gpu = n * gpu_hourly * HOURS_PER_MONTH
    else:
        on_demand = min(n, floor)          # the floor stays on-demand; everything above it is spot
        spot_n = n - on_demand
        monthly_gpu = (on_demand * gpu_hourly + spot_n * spot_hourly) * HOURS_PER_MONTH
    minutes = avg * 60 * HOURS_PER_MONTH
    gpu_per_min = monthly_gpu / minutes if minutes else float("inf")
    return {"peak": peak, "avg": avg, "gpus": n, "on_demand": on_demand, "spot": spot_n, "warm_floor": floor,
            "monthly_gpu": monthly_gpu, "minutes": minutes,
            "gpu_per_min": gpu_per_min, "total_per_min": gpu_per_min + telephony}


def overflow_minutes(peak: float, cap_per_gpu: float, gpus: int, avg: float) -> float:
    """Call-minutes in a month that `gpus` warm nodes cannot absorb (then priced at the paid rate).

    Capacity is `gpus * cap_per_gpu` simultaneous calls. The part of `peak` above that is not on
    these GPUs; `avg / peak` is the month's load factor, so overflow minutes are
    excess_calls * (avg / peak) * 60 * hours. A peak that fits returns 0. `avg` is what creates
    minutes — a peak with no average traffic overflows nothing.
    """
    if peak <= 0:
        return 0.0
    excess = max(0.0, peak - gpus * cap_per_gpu)
    return excess * (avg / peak) * 60.0 * HOURS_PER_MONTH


def break_even_calls(gpu_hourly: float, telephony: float, paid: float, gpus: int = 1) -> float:
    """Average simultaneous calls at which `gpus` GPUs cost the same per minute as the paid service."""
    room = paid - telephony
    return float("inf") if room <= 0 else gpus * gpu_hourly / 60.0 / room


def render(gpu_hourly: float, calls_per_gpu: float, telephony: float, paid: float,
           spot_hourly: float | None = None, warm_floor: int | None = 2) -> str:
    floor_n = max(0, int(warm_floor or 0))
    spot_txt = f", spot ${spot_hourly:.2f}/h above the warm floor" if spot_hourly is not None else ""
    out = [f"Inputs: GPU ${gpu_hourly:.2f}/h, {calls_per_gpu:g} calls per GPU at peak, telephony ${telephony:.4f}/min, "
           f"paid service ${paid:.3f}/min, {HOURS_PER_MONTH:g} h/month, 1 spare node, warm floor {floor_n}{spot_txt}.\n",
           "| Peak simultaneous calls | Average / peak | GPU nodes | GPU $/month | Call-minutes/month | GPU $/call-min | + phone line $/call-min | vs paid |",
           "|---|---|---|---|---|---|---|---|"]
    for peak in (10, 25, 50, 100, 200, 500):
        for ratio in (0.25, 0.5, 0.8):
            s = scenario(peak, peak * ratio, gpu_hourly=gpu_hourly, calls_per_gpu=calls_per_gpu, telephony=telephony,
                         spot_hourly=spot_hourly, warm_floor=warm_floor)
            vs = f"{paid / s['total_per_min']:.1f}x cheaper" if s["total_per_min"] < paid else f"{s['total_per_min'] / paid:.1f}x dearer"
            out.append(f"| {peak} | {ratio:.0%} | {s['gpus']} | ${s['monthly_gpu']:,.0f} | {s['minutes']:,.0f} | "
                       f"${s['gpu_per_min']:.4f} | ${s['total_per_min']:.4f} | {vs} |")
    be = break_even_calls(gpu_hourly, telephony, paid, gpus=2)
    out.append(f"\nWith the 2-node minimum, the paid service is cheaper until you average about {be:.1f} simultaneous calls "
               f"({be * 60 * HOURS_PER_MONTH:,.0f} call-minutes a month); one node would break even at "
               f"{break_even_calls(gpu_hourly, telephony, paid):.1f}.")
    # Largest row in the table: what the warm floor alone cannot hold is priced at --paid, not as a free GPU.
    top = scenario(500, 250, gpu_hourly=gpu_hourly, calls_per_gpu=calls_per_gpu, telephony=telephony,
                   spot_hourly=spot_hourly, warm_floor=warm_floor)
    ov = overflow_minutes(500, calls_per_gpu, floor_n, 250)
    out.append(f"Warm floor {floor_n}, on-demand nodes {top['on_demand']}, spot nodes {top['spot']}, "
               f"overflow minutes the warm floor cannot absorb {ov:,.0f} priced at the paid rate ${paid:.3f}/min "
               f"(${ov * paid:,.0f}/month). This does not remove idle-GPU cost or cold-start time.")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gpu-hourly", type=float, default=2.86)
    ap.add_argument("--calls-per-gpu", type=float, default=24)
    ap.add_argument("--telephony", type=float, default=0.0085)
    ap.add_argument("--paid", type=float, default=0.05)
    ap.add_argument("--spot-hourly", type=float, default=None)
    ap.add_argument("--warm-floor", type=int, default=2)
    a = ap.parse_args()
    print(render(a.gpu_hourly, a.calls_per_gpu, a.telephony, a.paid,
                 spot_hourly=a.spot_hourly, warm_floor=a.warm_floor))
