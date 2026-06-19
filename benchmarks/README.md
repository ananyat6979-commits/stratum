# STRATUM Benchmark Methodology

## Core Principle

Every benchmark result in `results/` is a falsifiable claim with a stated
null hypothesis, coordinated omission correction, bootstrap confidence
intervals, and hardware fingerprint. A number without a confidence interval
is an anecdote. A benchmark without a stated hypothesis is unfalsifiable.

## Coordinated Omission Correction

Standard load generators (wrk, ab, locust defaults) exhibit **coordinated
omission**: when the server is slow, the generator slows its issue rate to
match, causing slow responses to be undersampled. The measured P99 appears
lower than the actual P99 under sustained load.

STRATUM's harness corrects for this using Gil Tene's method: for each
response that arrives late (later than one inter-arrival interval), we
synthesize phantom measurements for all the requests that *would* have been
issued during the slow period. This produces the P99 that clients actually
experience, not the P99 that a naive benchmark would report.

**STRATUM does not publish uncorrected latency numbers.**

Reference: Tene (2015), "How NOT to measure latency",
https://www.infoq.com/presentations/latency-pitfalls

## Workload Model

Requests arrive as a Poisson process (exponentially distributed
inter-arrival times). This is the correct model for web/API traffic and
produces realistic variance in load. Fixed inter-arrival time generators
underestimate variance and produce optimistic tail latency measurements.

## Statistical Reporting

Every latency summary includes:
- Point estimate (sample percentile)
- 95% bootstrap confidence interval (non-parametric, no normality assumption)
- Sample size (number of CO-corrected measurements)

Every A/B comparison includes:
- Welch's t-test (not Student's, no equal-variance assumption)
- Bootstrap CI on the absolute difference
- Bootstrap CI on the relative difference (%)
- Cohen's d effect size with conventional interpretation
- p-value with explicitly stated alpha and null hypothesis

## Result Files

Each benchmark run produces two files in `benchmarks/results/`:

| File | Contents |
|------|----------|
| `<name>_<date>_<run_id>.parquet` | One row per CO-corrected measurement, with metadata columns for reproducibility |
| `<name>_<date>_<run_id>_meta.json` | Human-readable summary: config, percentiles with CIs, hardware fingerprint, git SHA |

Parquet files use Snappy compression. They can be read with DuckDB,
pandas, or any Parquet reader.

## Reproducing a Result

1. Check out the git SHA from the `meta.json` file
2. Start `stratum-gateway` on `127.0.0.1:8080`
3. Run the scenario with the same config

```powershell
# Install harness dependencies
cd benchmarks\harness
uv pip install pyarrow numpy scipy pyyaml httpx

# Run the baseline scenario
python runner.py run `
    --config ..\scenarios\gateway_baseline.yaml `
    --output ..\results\
```

## Benchmark Phases

| Phase | Benchmark | Compares |
|-------|-----------|----------|
| 0/1 | `gateway_baseline_round_robin` | Absolute gateway overhead, no inference |
| 2 | `routing_semantic_vs_roundrobin` | Semantic router cache hit rate + latency |
| 3 | `scheduler_sjcf_vs_fcfs` | SJCF vs FCFS on Zipfian length distributions |
| 4+ | Per-chaos-mode recovery benchmarks | MTTF/MTTR per failure mode |

## What These Benchmarks Do NOT Measure (Yet)

- Inference latency (no model running in Phase 0/1)
- Multi-worker load balancing effectiveness (no workers)
- KV cache hit rate improvement (no FAISS index)
- NUMA memory bandwidth effects (no NUMA hardware / emulation configured)

These are explicitly deferred and documented here so the scope of each
benchmark's claims is unambiguous.
