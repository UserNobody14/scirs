// Cross-crate integration tests for SciRS2 v0.2.0
// Tests module interactions, data flow, and API compatibility

// Common utilities and test helpers
pub mod common;
pub mod fixtures;

// Cross-module integration tests
pub mod neural_optimize;
pub mod fft_signal;
pub mod sparse_linalg;
pub mod ndimage_vision;
pub mod stats_datasets;

// Performance and memory integration tests
pub mod performance;
