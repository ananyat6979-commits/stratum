//! SLA class assignment for incoming requests.
//!
//! Every request is assigned an [`SlaClass`] at ingress based on the
//! `Authorization` header tier. SLA class drives:
//! - Priority queuing in the router (REALTIME > INTERACTIVE > BATCH)
//! - Routing score weighting (higher SLA → stronger pressure avoidance)
//! - Token bucket allocation (separate buckets per SLA class)
//!
//! # Assignment Rules
//! | Authorization prefix | SLA class   |
//! |----------------------|-------------|
//! | `Bearer rt-`         | REALTIME    |
//! | `Bearer ia-`         | INTERACTIVE |
//! | `Bearer bt-` or none | BATCH       |
//!
//! The prefix convention is internal. External clients are issued tokens
//! with the appropriate prefix by the auth service. The gateway treats
//! the prefix as the authoritative SLA signal — no other header is consulted.

/// SLA class for a request. Determines priority and routing behaviour.
///
/// Ordered: REALTIME > INTERACTIVE > BATCH.
/// The `u8` representation is used for priority queue ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum SlaClass {
    Batch = 0,
    Interactive = 1,
    Realtime = 2,
}

impl SlaClass {
    /// Returns the display name used in metrics labels and log fields.
    /// Kept lowercase and stable — changing these breaks dashboards.
    pub fn as_str(&self) -> &'static str {
        match self {
            SlaClass::Realtime => "realtime",
            SlaClass::Interactive => "interactive",
            SlaClass::Batch => "batch",
        }
    }
}

/// Assigns an SLA class from the value of the `Authorization` header.
///
/// # Arguments
/// * `auth_header` - The raw value of the `Authorization` header, or `None`
///   if the header is absent. A missing header is treated as BATCH — no error.
///
/// # Examples
/// ```
/// use stratum_gateway::sla::{SlaClass, assign_sla_class};
///
/// assert_eq!(assign_sla_class(Some("Bearer rt-abc123")), SlaClass::Realtime);
/// assert_eq!(assign_sla_class(Some("Bearer ia-abc123")), SlaClass::Interactive);
/// assert_eq!(assign_sla_class(Some("Bearer bt-abc123")), SlaClass::Batch);
/// assert_eq!(assign_sla_class(None), SlaClass::Batch);
/// ```
pub fn assign_sla_class(auth_header: Option<&str>) -> SlaClass {
    let Some(header) = auth_header else {
        return SlaClass::Batch;
    };

    // Reject anything that doesn't start with "Bearer " — malformed headers
    // must never accidentally receive elevated priority.
    let Some(token) = header.strip_prefix("Bearer ") else {
        return SlaClass::Batch;
    };

    if token.starts_with("rt-") {
        SlaClass::Realtime
    } else if token.starts_with("ia-") {
        SlaClass::Interactive
    } else {
        SlaClass::Batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realtime_token_prefix_assigns_realtime() {
        assert_eq!(
            assign_sla_class(Some("Bearer rt-abc123")),
            SlaClass::Realtime
        );
    }

    #[test]
    fn interactive_token_prefix_assigns_interactive() {
        assert_eq!(
            assign_sla_class(Some("Bearer ia-xyz789")),
            SlaClass::Interactive
        );
    }

    #[test]
    fn batch_token_prefix_assigns_batch() {
        assert_eq!(assign_sla_class(Some("Bearer bt-def456")), SlaClass::Batch);
    }

    #[test]
    fn missing_header_assigns_batch() {
        assert_eq!(assign_sla_class(None), SlaClass::Batch);
    }

    #[test]
    fn unrecognized_token_assigns_batch_not_error() {
        // Unknown tokens must never cause errors — fail-safe to BATCH
        assert_eq!(
            assign_sla_class(Some("Bearer unknown-token")),
            SlaClass::Batch
        );
    }

    #[test]
    fn missing_bearer_prefix_assigns_batch() {
        // Malformed header (no "Bearer " prefix) → BATCH
        assert_eq!(
            assign_sla_class(Some("rt-abc123")),
            SlaClass::Batch,
            "token without Bearer prefix should not get REALTIME"
        );
    }

    #[test]
    fn sla_class_ordering_is_correct() {
        // REALTIME must be highest priority for the priority queue to work
        assert!(SlaClass::Realtime > SlaClass::Interactive);
        assert!(SlaClass::Interactive > SlaClass::Batch);
    }

    #[test]
    fn as_str_returns_stable_labels() {
        // These strings are used as Prometheus label values — must never change
        assert_eq!(SlaClass::Realtime.as_str(), "realtime");
        assert_eq!(SlaClass::Interactive.as_str(), "interactive");
        assert_eq!(SlaClass::Batch.as_str(), "batch");
    }
}
