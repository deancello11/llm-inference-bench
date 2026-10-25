"""Phase 0 / Phase 3 gate: is this GPU instance usable for energy benchmarking?

    python scripts/check_gpu.py            # ~40 s

Checks: full GPU (not MIG), NVML power is real and *changes* under load, the energy counter
exists and agrees with integrated power samples. Run this on a rented pod BEFORE building
anything else; if it fails, change provider.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench.gpu import GpuMonitor  # noqa: E402


def main() -> int:
    import torch

    mon = GpuMonitor(0)
    mon.start()
    info = mon.info()
    print("GPU info:", info)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    check("not a MIG slice", not info.get("mig_enabled"))
    check("NVML energy counter supported", info["energy_counter"])

    print("measuring idle (10 s)...")
    idle = mon.measure_idle_power_w(10)
    print(f"idle power: {idle:.1f} W")

    print("running matmul load (20 s)...")
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    w = mon.begin_window()
    t_end = time.perf_counter() + 20
    while time.perf_counter() < t_end:
        for _ in range(10):
            a @ b
        torch.cuda.synchronize()
    r = mon.end_window(w)
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})

    check("power rises under load", r["power_mean_w"] > idle * 1.5,
          f"(idle {idle:.0f} W -> load {r['power_mean_w']:.0f} W)")
    if r["energy_counter_j"] is not None:
        rel = abs(r["energy_counter_j"] - r["energy_integrated_j"]) / r["energy_counter_j"]
        check("counter vs integrated power agree within 5%", rel < 0.05, f"(diff {rel*100:.1f}%)")
    check("no thermal throttling during load test", not r["thermal_throttled"],
          f"(throttle mask 0x{r['throttle_mask']:x}, max {r['temp_max_c']:.0f} C)")
    mon.stop()
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED - consider another provider")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
