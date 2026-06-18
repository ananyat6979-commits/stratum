//! Transcoding between the OpenAI-compatible wire format and STRATUM's
//! internal `InferenceRequest` proto.
//!
//! External clients send OpenAI-style JSON. Internally, every service
//! after the gateway communicates via the generated `InferenceRequest`
//! proto (see `inference.proto`). This module is the single point where
//! that translation happens — no other code should construct
//! `InferenceRequest` by hand.

use crate::signing::{compute_replay_key, hash_body};
use crate::sla::SlaClass as InternalSlaClass;

// Generated proto types live in $OUT_DIR; included once here and
// re-exported so the rest of the crate can `use crate::proto::InferenceRequest`.
include!(concat!(env!("OUT_DIR"), "/stratum.v1.rs"));

/// Minimal OpenAI-compatible request shape. Only the fields STRATUM
/// actually uses are modeled — we are not a full OpenAI API surface.
#[derive(Debug, serde::Deserialize)]
pub struct OpenAiCompatRequest {
    #[allow(dead_code)] // retained for future model-based routing; unused today
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: i32,
}

#[derive(Debug, serde::Deserialize)]
pub struct ChatMessage {
    #[allow(dead_code)] // role is not yet used for routing; reserved for Phase 2
    pub role: String,
    pub content: String,
}

fn default_max_tokens() -> i32 {
    256
}

/// Converts the internal hand-written [`InternalSlaClass`] (used for
/// priority-queue ordering in the router) to the generated proto
/// [`SlaClass`] (used for wire transmission).
///
/// These are deliberately separate types: the internal enum's variant
/// order encodes priority (`Batch < Interactive < Realtime`) via
/// `#[repr(u8)]` discriminants, which the proto enum's wire-stable
/// numbering must not be coupled to. Proto enum numbering is a wire
/// contract; internal enum ordering is an implementation detail of the
/// priority queue. Conflating them would mean a wire-format change
/// (adding a new SLA tier) could silently break priority ordering.
fn to_proto_sla_class(internal: InternalSlaClass) -> SlaClass {
    match internal {
        InternalSlaClass::Realtime => SlaClass::Realtime,
        InternalSlaClass::Interactive => SlaClass::Interactive,
        InternalSlaClass::Batch => SlaClass::Batch,
    }
}

/// Extracts a single prompt string from an OpenAI-style message list.
///
/// STRATUM's worker layer (Ollama/vLLM) takes a flat prompt string, not
/// a structured message list. This concatenates all message contents
/// with newlines. This is a placeholder strategy — proper chat templating
/// (role-aware formatting per model) is deferred to the worker layer in
/// a later phase; the gateway's job is request routing, not prompt
/// engineering.
fn extract_prompt(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Transcodes an OpenAI-compatible request into STRATUM's internal
/// `InferenceRequest` proto.
///
/// # Arguments
/// * `raw_body` - The raw request body bytes, as received over HTTP.
///   Used to compute `replay_key` — the body must be hashed before
///   any parsing, so the replay key reflects exactly what the client sent.
/// * `parsed` - The deserialized OpenAI-compatible request.
/// * `auth_header` - Raw `Authorization` header value, passed through
///   to [`crate::sla::assign_sla_class`].
/// * `ingress_timestamp_ns` - Wall clock at ingress. Caller-supplied
///   (not computed internally) so tests can inject deterministic timestamps.
/// * `ingress_node_id` - This gateway instance's node identifier.
///
/// # Determinism note
/// `replay_key` is computed from `raw_body`, not from `parsed` — this
/// guarantees the key reflects the exact bytes received, independent of
/// how `serde` might reformat or reorder fields during deserialization.
pub fn to_inference_request(
    raw_body: &[u8],
    parsed: &OpenAiCompatRequest,
    auth_header: Option<&str>,
    ingress_timestamp_ns: i64,
    ingress_node_id: &str,
) -> InferenceRequest {
    let body_hash = hash_body(raw_body);
    let replay_key = compute_replay_key(ingress_timestamp_ns, &body_hash, ingress_node_id);
    let internal_sla = crate::sla::assign_sla_class(auth_header);

    InferenceRequest {
        replay_key,
        sla_class: to_proto_sla_class(internal_sla) as i32,
        ingress_timestamp_ns,
        prompt: extract_prompt(&parsed.messages),
        max_tokens: parsed.max_tokens,
        session_id: None,
        experiment_treatment: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> OpenAiCompatRequest {
        OpenAiCompatRequest {
            model: "phi3:mini".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hello world".to_string(),
            }],
            max_tokens: 256,
        }
    }

    #[test]
    fn transcodes_prompt_from_single_message() {
        let req = sample_request();
        let result = to_inference_request(b"raw body", &req, None, 1000, "node-0");
        assert_eq!(result.prompt, "hello world");
    }

    #[test]
    fn concatenates_multiple_messages_with_newlines() {
        let req = OpenAiCompatRequest {
            model: "phi3:mini".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "be helpful".to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                },
            ],
            max_tokens: 256,
        };
        let result = to_inference_request(b"raw body", &req, None, 1000, "node-0");
        assert_eq!(result.prompt, "be helpful\nhello");
    }

    #[test]
    fn max_tokens_passes_through_unchanged() {
        let req = sample_request();
        let result = to_inference_request(b"raw body", &req, None, 1000, "node-0");
        assert_eq!(result.max_tokens, 256);
    }

    #[test]
    fn missing_auth_header_maps_to_batch_sla() {
        let req = sample_request();
        let result = to_inference_request(b"raw body", &req, None, 1000, "node-0");
        assert_eq!(result.sla_class, SlaClass::Batch as i32);
    }

    #[test]
    fn realtime_auth_header_maps_to_realtime_sla() {
        let req = sample_request();
        let result =
            to_inference_request(b"raw body", &req, Some("Bearer rt-abc123"), 1000, "node-0");
        assert_eq!(result.sla_class, SlaClass::Realtime as i32);
    }

    #[test]
    fn replay_key_is_deterministic_for_identical_inputs() {
        let req = sample_request();
        let key1 = to_inference_request(b"identical body", &req, None, 1000, "node-0").replay_key;
        let key2 = to_inference_request(b"identical body", &req, None, 1000, "node-0").replay_key;
        assert_eq!(key1, key2);
    }

    #[test]
    fn replay_key_changes_with_raw_body_not_parsed_fields() {
        // Two different raw bodies, but same parsed result (simulating
        // whitespace/field-order differences in the original JSON) must
        // produce different replay keys, since signing is over raw bytes.
        let req = sample_request();
        let key1 = to_inference_request(b"body variant A", &req, None, 1000, "node-0").replay_key;
        let key2 = to_inference_request(b"body variant B", &req, None, 1000, "node-0").replay_key;
        assert_ne!(key1, key2);
    }

    #[test]
    fn session_id_and_experiment_treatment_default_to_none() {
        let req = sample_request();
        let result = to_inference_request(b"raw body", &req, None, 1000, "node-0");
        assert_eq!(result.session_id, None);
        assert_eq!(result.experiment_treatment, None);
    }
}
