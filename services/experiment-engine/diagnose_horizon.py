"""
One-off diagnostic, not part of the test suite, checking whether
should_reject_null ever fires under the null hypothesis given a long
horizon, and if so, how early or late those rejections happen relative
to test_msprt_type1_error.py's 500-step cutoff.

Run as a script, not pasted into a REPL: the REPL echoes every
expression's return value at each line, which produced 800 lines of
noise instead of a clean summary the one time this was tried
interactively. This version prints only explicit summary lines.

Also logs the running e_value at fixed checkpoints for the FIRST
simulation specifically, since a live REPL run showed what looked like
smooth, monotonic decay toward zero rather than the noisy up-and-down
wandering a martingale should show under repeated independent draws,
worth confirming or refuting directly against a clean single trace
before drawing any conclusion from the aggregate rejection rate alone.
"""

from __future__ import annotations

import random

from stratum_experiment.msprt import MSPRTConfig, MSPRTState

N_SIMS = 500
HORIZON = 3000
CHECKPOINTS = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 3000]
SEED = 20260811


def main() -> None:
    config = MSPRTConfig(alpha=0.05, sigma_squared=1.0, tau_squared=0.25)
    rng = random.Random(SEED)
    per_arm_sd = (config.sigma_squared / 2.0) ** 0.5

    rejection_steps: list[int] = []
    first_trace: dict[int, float] = {}

    for sim_idx in range(N_SIMS):
        state = MSPRTState(config=config)
        for step in range(1, HORIZON + 1):
            treatment = rng.gauss(0.0, per_arm_sd)
            control = rng.gauss(0.0, per_arm_sd)
            state.update(treatment, control)

            if sim_idx == 0 and step in CHECKPOINTS:
                first_trace[step] = state.e_value

            if state.should_reject_null:
                rejection_steps.append(step)
                break

    print(f"=== {N_SIMS} simulations, {HORIZON}-step horizon, alpha=0.05 (threshold={config.rejection_threshold}) ===")
    print(f"total rejections: {len(rejection_steps)} / {N_SIMS} = {len(rejection_steps)/N_SIMS:.4f}")
    if rejection_steps:
        within_500 = sum(1 for s in rejection_steps if s <= 500)
        print(f"of those, rejected within first 500 steps: {within_500} ({within_500/len(rejection_steps):.1%})")
        print(f"rejection step min/median/max: {min(rejection_steps)} / {sorted(rejection_steps)[len(rejection_steps)//2]} / {max(rejection_steps)}")
    else:
        print("zero rejections across all simulations at this horizon")

    print()
    print(f"=== first simulation's e_value at fixed checkpoints (seed={SEED}) ===")
    for step in CHECKPOINTS:
        if step in first_trace:
            print(f"  step {step:>5}: e_value = {first_trace[step]:.6f}")


if __name__ == "__main__":
    main()