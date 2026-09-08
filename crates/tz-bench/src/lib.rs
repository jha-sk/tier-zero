//! Load generation and latency accounting.
//!
//! The one thing this module exists to get right is **coordinated omission**.
//!
//! A closed-loop load generator sends a request, waits for the response, then
//! sends the next. When the system slows down, the generator slows down with
//! it, so the slow period receives *fewer* samples than a fast one. The
//! measured p99 then describes a load level the system never actually
//! experienced, and it is biased optimistic exactly when the system is
//! struggling — the case the number exists to describe.
//!
//! The fix is an open model: requests are scheduled against a fixed arrival
//! rate decided in advance, and a request that starts late is charged for the
//! waiting it did. Latency is measured from **intended** send time, not from
//! actual send time.

pub mod openloop;

pub use openloop::{LoadReport, LoadTest, Schedule};
