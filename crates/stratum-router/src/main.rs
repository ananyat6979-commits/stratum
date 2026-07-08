//! Manual verification binary for HttpSignalsProvider.
//!
//! Run cache-oracle first (uvicorn), register a worker via curl, then
//! run this binary to confirm the polling loop actually fetches and
//! caches real signals from a live process -- not just mocked logic.
//!
//! This is a temporary manual-test entrypoint, not the production
//! router binary. It will be replaced once stratum-gateway integrates
//! stratum-router directly.

use std::time::Duration;

use stratum_router::http_signals_provider::HttpSignalsProvider;
use stratum_router::semantic_router::WorkerSignalsProvider;

#[tokio::main]
async fn main() {
    let base_url = "http://127.0.0.1:8001";
    let poll_interval = Duration::from_secs(2);
    let max_staleness = Duration::from_secs(10);

    println!("stratum-router manual verification");
    println!("Polling cache-oracle at {base_url} every {poll_interval:?}");
    println!(
        "(make sure cache-oracle is running: uv run uvicorn stratum_oracle.api:app --port 8001)"
    );
    println!();

    let provider = HttpSignalsProvider::new(base_url, poll_interval, max_staleness);

    // Give the first poll a moment to complete before checking results
    tokio::time::sleep(Duration::from_secs(1)).await;

    for i in 0..10 {
        let signals = provider.signals_for_workers(&["worker-0"]);
        println!("[{i}] {:?}", signals[0]);
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}
