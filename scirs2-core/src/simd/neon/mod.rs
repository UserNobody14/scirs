//! ARM NEON SIMD optimizations for mobile platforms
//!
//! This module provides ARM NEON-specific optimizations for iOS and Android.
//! All operations are designed with mobile constraints in mind:
//! - Battery efficiency (minimal power consumption)
//! - Memory efficiency (reduced allocations)
//! - Thermal awareness (prevent overheating)
//!
//! ## Architecture Support
//!
//! - ARMv8-A (AArch64): Full NEON support
//! - ARMv7-A: NEON support with some limitations
//!
//! ## Modules
//!
//! - [`basic`]: Basic NEON arithmetic operations
//! - [`matrix`]: Matrix operations optimized for NEON
//! - [`activation`]: Neural network activation functions
//! - [`mobile`]: Mobile-specific optimizations

pub mod activation;
pub mod basic;
pub mod matrix;
pub mod mobile;

// Re-export commonly used functions
pub use basic::{
    neon_add_f32, neon_add_f64, neon_div_f32, neon_dot_f32, neon_dot_f64, neon_mul_f32,
    neon_mul_f64, neon_sub_f32, neon_sub_f64,
};

pub use matrix::{neon_gemm_f32, neon_gemm_f64, neon_gemv_f32, neon_gemv_f64};

pub use activation::{
    neon_gelu_f32, neon_leaky_relu_f32, neon_relu_f32, neon_sigmoid_f32, neon_tanh_f32,
};

pub use mobile::{
    neon_dot_battery_optimized, neon_gemm_battery_optimized, neon_gemm_thermal_aware, BatteryMode,
    MobileOptimizer, ThermalState,
};

/// Check if NEON is available at runtime
#[inline]
pub fn is_neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_detection() {
        let available = is_neon_available();
        #[cfg(target_arch = "aarch64")]
        {
            // On AArch64, NEON should always be available
            assert!(available, "NEON should be available on AArch64");
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            assert!(!available, "NEON should not be available on non-ARM");
        }
    }
}
