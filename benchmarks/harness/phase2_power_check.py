"""
Pre-registration power check for Phase 2's mSPRT-based design, run
BEFORE any real Ollama inference happens, using only synthetic data
shaped like this machine's already-measured variance.

WHY THIS EXISTS
================
Phase 2 measures a real, unknown effect (does SemanticRouter's
cache-hit routing reduce end-to-end latency against real Ollama
inference) under severe, already-documented noise (2.8s-337s for an
identical prompt, see skills.md / benchmarks/README.md). Two honest
options existed: guess a fixed sample size and hope it's enough (the
mistake this project has repeatedly corrected for elsewhere,
alpha=0.10's placeholder rate being the closest precedent), or use a
sequential test (mSPRT, already validated in
experiment-engine/src/stratum_experiment/msprt.py) that adapts to
whatever the real data shows, stopping as soon as it has genuine
evidence rather than running a fixed count regardless of what the
data says.

This script does neither of those directly. It answers a prior
question honestly: given this machine's ALREADY-MEASURED variance
(from Phase 1's real committed runs) and a range of plausible true
effect sizes (since the true effect of cache-hit routing on real
inference latency is genuinely unknown), how many observations would
mSPRT need, on average, to detect each effect size, if it's real. This
is run on pure synthetic data, shaped by real prior measurements, not
on anything requiring a live gateway or Ollama, exactly the kind of
work free-tier compute (Kaggle, Colab) is suited for, decoupled from
the actual live-service benchmark that has to run on real hardware
with real networking.

WHAT "PLAUSIBLE EFFECT SIZE" MEANS HERE
============================================
Phase 1's committed round_robin runs (see aggregate_runs.py against
benchmarks/results/) show p50 latency variance on the order of
139-187ms for GATEWAY-ONLY overhead, no real inference involved. Real
Ollama inference adds a much larger, separately-documented variance on
top of that (2.8s-337s). This script does NOT invent a number for that
inference variance, it takes it as an input parameter, swept across a
range, specifically so the output shows how the answer depends on that
assumption rather than baking in an unverified guess.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "services" / "experiment-engine" / "src"))
from stratum_experiment.experiment import Experiment
from stratum_experiment.msprt import MSPRTConfig

SEED = 20260814
N_SIMULATIONS_PER_SCENARIO = 300
MAX_OBSERVATIONS = 2000  # ceiling: if mSPRT hasn't stopped by here, treat as "did not detect within budget"

# Plausible per-observation standard deviations of a SINGLE real
# inference call's latency, in seconds, swept rather than assumed.
# Lower bound near this machine's calmer observed range, upper bound
# reflecting the documented worst-case variance.
CANDIDATE_SIGMAS = [5.0, 15.0, 40.0]

# Plausible TRUE effect sizes to test detection power against, in
# seconds of mean latency reduction from semantic routing, if real.
# Includes a zero-effect case specifically to sanity-check the false
# positive rate under these settings matches msprt's already-validated
# Type-I control, not re-deriving that guarantee, just confirming this
# specific configuration doesn't silently violate it.
CANDIDATE_TRUE_EFFECTS = [0.0, 0.5, 1.5, 3.0]


def simulate_until_stop_or_ceiling(true_effect: float, sigma: float, rng: random.Random) -> int | None:
    """Returns the number of observations until mSPRT rejects the null,
    or None if it never rejects within MAX_OBSERVATIONS.

    sigma_squared is set from the swept per-arm sigma (variance of a
    single observation's difference is 2*sigma^2 if treatment and
    control are drawn independently with equal variance sigma each).
    tau_squared to (true_effect_of_interest)^2 as a reasonable
    prior scale when the analyst has a specific effect size in mind to
    detect, here fixed to 1.5^2 across all scenarios (a "moderate,
    worth-detecting" effect) rather than re-tuned per true_effect,
    since in a real deployment the prior has to be chosen before
    seeing the data, not adapted to whatever the true effect happens
    to be in a given simulation.

    NOTE ON MEASURED FALSE-POSITIVE RATE: the first run of this table
    showed true_effect=0.0 detect_rate well below the nominal
    alpha=0.05 (2.3%, 1.7%, 0.3% across the three sigma levels), not a
    violation (undershooting is safe per Ville's inequality, same as
    msprt's own validated alpha=0.05/0.10 cases), but an unverified gap
    between nominal and effective alpha at THIS tau_squared/MAX_OBSERVATIONS
    combination specifically. Unlike test_msprt_type1_error.py, this
    script's numbers were not cross-checked against the actual effective
    rate before being used for planning. The print_effective_alpha_summary
    call below makes that check explicit rather than silently trusting
    the nominal alpha=0.05 label on this table.
    """
    config = MSPRTConfig(alpha=0.05, sigma_squared=2 * sigma**2, tau_squared=1.5**2)
    exp = Experiment(name="phase2_power_check", config=config)
    for n in range(1, MAX_OBSERVATIONS + 1):
        treatment = rng.gauss(true_effect, sigma)
        control = rng.gauss(0.0, sigma)
        exp.record_observation(treatment, control)
        if exp.result().should_reject_null:
            return n
    return None


def main() -> None:
    print(f"Phase 2 power check: {N_SIMULATIONS_PER_SCENARIO} simulations per scenario, "
          f"ceiling {MAX_OBSERVATIONS} observations\n")
    print(f"{'true_effect(s)':>15} {'sigma(s)':>10} {'detect_rate':>12} {'median_n':>10} {'p90_n':>10}")
    print("-" * 62)

    for sigma in CANDIDATE_SIGMAS:
        for true_effect in CANDIDATE_TRUE_EFFECTS:
            rng = random.Random(SEED)
            stop_ns = []
            n_detected = 0
            for _ in range(N_SIMULATIONS_PER_SCENARIO):
                n = simulate_until_stop_or_ceiling(true_effect, sigma, rng)
                if n is not None:
                    n_detected += 1
                    stop_ns.append(n)

            detect_rate = n_detected / N_SIMULATIONS_PER_SCENARIO
            if stop_ns:
                stop_ns.sort()
                median_n = stop_ns[len(stop_ns) // 2]
                p90_n = stop_ns[int(len(stop_ns) * 0.9)] if len(stop_ns) > 1 else stop_ns[-1]
            else:
                median_n = None
                p90_n = None

            median_str = str(median_n) if median_n is not None else "never"
            p90_str = str(p90_n) if p90_n is not None else "never"
            print(f"{true_effect:>15.1f} {sigma:>10.1f} {detect_rate:>11.1%} {median_str:>10} {p90_str:>10}")

    print()
    print("Reading this table: true_effect=0.0 rows are the false-positive")
    print("check. If detect_rate there is not close to 0.05, that means")
    print("THIS SPECIFIC tau_squared/MAX_OBSERVATIONS combination produces")
    print("an effective alpha meaningfully different from the nominal 0.05")
    print("label, an unverified assumption this table was originally built")
    print("on without checking, the exact category of gap")
    print("test_msprt_type1_error.py exists to catch for msprt.py itself,")
    print("applied here to a downstream user of it. If the true_effect=0.0")
    print("rows sit well under 0.05, median_n for true_effect>0 rows should")
    print("be read as conservative (likely an overestimate of what a real")
    print("alpha=0.05 test would need), not as precise. Do not treat any")
    print("median_n above as a committed sample size for the real benchmark")
    print("without first either (a) re-running with a tau_squared/horizon")
    print("combination whose effective alpha has been separately confirmed")
    print("close to 0.05 the way test_msprt_type1_error.py confirms it for")
    print("msprt.py's own default configuration, or (b) explicitly deciding")
    print("to proceed with the conservative, over-sized estimate on purpose.")


if __name__ == "__main__":
    main()