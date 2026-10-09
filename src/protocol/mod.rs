//! Protocol module for Jev Wire Protocol specification.

pub mod request;
pub mod response;

pub use request::{Question, SystemOneRequest, MAX_OPTIONS};
pub use response::{round_to, Answer, SystemOneResponse, Usage};
