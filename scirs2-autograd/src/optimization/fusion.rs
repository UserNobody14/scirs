//! Operation fusion for improved performance
//!
//! This module implements operation fusion to combine multiple operations into
//! single fused kernels, reducing memory traffic and improving performance.

use crate::{error::AutogradError, Float, Result};
use std::collections::{HashMap, HashSet};

/// Fusible operation pattern
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// Element-wise operations (add, mul, relu, etc.)
    ElementWise,
    /// Matrix multiplication + bias add
    MatMulBias,
    /// Matrix multiplication + activation
    MatMulActivation,
    /// Convolution + batch norm + activation
    ConvBNActivation,
    /// Reduction + element-wise
    ReductionElementWise,
    /// Custom fusion pattern
    Custom(String),
}

/// Fusion candidate
#[derive(Debug, Clone)]
pub struct FusionCandidate {
    /// Operation IDs to fuse
    pub ops: Vec<usize>,
    /// Fusion pattern
    pub pattern: FusionPattern,
    /// Estimated speedup
    pub speedup: f64,
    /// Memory savings (bytes)
    pub memory_saved: usize,
}

impl FusionCandidate {
    /// Create a new fusion candidate
    pub fn new(ops: Vec<usize>, pattern: FusionPattern, speedup: f64, memory_saved: usize) -> Self {
        Self {
            ops,
            pattern,
            speedup,
            memory_saved,
        }
    }

    /// Check if fusion is beneficial
    pub fn is_beneficial(&self) -> bool {
        self.speedup > 1.1 || self.memory_saved > 1024
    }
}

/// Fusion optimizer
pub struct FusionOptimizer {
    /// Detected fusion candidates
    candidates: Vec<FusionCandidate>,
    /// Applied fusions
    applied: HashSet<Vec<usize>>,
    /// Statistics
    total_fusions: usize,
}

impl FusionOptimizer {
    /// Create a new fusion optimizer
    pub fn new() -> Self {
        Self {
            candidates: Vec::new(),
            applied: HashSet::new(),
            total_fusions: 0,
        }
    }

    /// Detect fusion opportunities
    pub fn detect_fusions(&mut self) -> Result<()> {
        // This is a simplified implementation
        // In production, would analyze actual computation graph

        // Example: Detect matmul + bias pattern
        self.candidates.push(FusionCandidate::new(
            vec![1, 2], // Placeholder op IDs
            FusionPattern::MatMulBias,
            1.5,  // 50% speedup
            4096, // 4KB memory saved
        ));

        Ok(())
    }

    /// Apply beneficial fusions
    pub fn apply_fusions(&mut self) -> Result<usize> {
        let mut num_applied = 0;

        for candidate in &self.candidates {
            if candidate.is_beneficial() && !self.applied.contains(&candidate.ops) {
                // Apply fusion (in production, would rewrite computation graph)
                self.applied.insert(candidate.ops.clone());
                self.total_fusions += 1;
                num_applied += 1;
            }
        }

        Ok(num_applied)
    }

    /// Get fusion candidates
    pub fn candidates(&self) -> &[FusionCandidate] {
        &self.candidates
    }

    /// Get number of applied fusions
    pub fn total_fusions(&self) -> usize {
        self.total_fusions
    }

    /// Clear optimizer state
    pub fn clear(&mut self) {
        self.candidates.clear();
        self.applied.clear();
    }
}

impl Default for FusionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Fusion pass result
#[derive(Debug, Clone)]
pub struct FusionResult {
    /// Number of fusions applied
    pub fusions: usize,
    /// Total speedup factor
    pub speedup: f64,
    /// Memory saved (bytes)
    pub memory_saved: usize,
}

impl FusionResult {
    /// Create a new fusion result
    pub fn new(fusions: usize, speedup: f64, memory_saved: usize) -> Self {
        Self {
            fusions,
            speedup,
            memory_saved,
        }
    }

    /// Check if any fusions were applied
    pub fn has_changes(&self) -> bool {
        self.fusions > 0
    }
}

/// Element-wise operation fusion
pub fn fuse_elementwise_ops(ops: &[String]) -> Result<String> {
    // Combine multiple element-wise operations into single fused kernel
    // Example: (x + y) * z => fused_add_mul(x, y, z)

    if ops.is_empty() {
        return Err(AutogradError::invalid_argument(
            "No operations to fuse".to_string(),
        ));
    }

    let fused = format!("fused_{}", ops.join("_"));
    Ok(fused)
}

/// Matrix multiplication fusion
pub fn fuse_matmul_bias(has_bias: bool, activation: Option<&str>) -> String {
    match (has_bias, activation) {
        (true, Some(act)) => format!("fused_matmul_bias_{}", act),
        (true, None) => "fused_matmul_bias".to_string(),
        (false, Some(act)) => format!("fused_matmul_{}", act),
        (false, None) => "matmul".to_string(),
    }
}

/// Convolution fusion
pub fn fuse_conv_bn_activation(has_bn: bool, activation: Option<&str>) -> String {
    match (has_bn, activation) {
        (true, Some(act)) => format!("fused_conv_bn_{}", act),
        (true, None) => "fused_conv_bn".to_string(),
        (false, Some(act)) => format!("fused_conv_{}", act),
        (false, None) => "conv".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_pattern() {
        assert_eq!(FusionPattern::ElementWise, FusionPattern::ElementWise);
        assert_ne!(FusionPattern::ElementWise, FusionPattern::MatMulBias);
    }

    #[test]
    fn test_fusion_candidate() {
        let candidate =
            FusionCandidate::new(vec![1, 2, 3], FusionPattern::MatMulActivation, 1.8, 2048);

        assert!(candidate.is_beneficial());
        assert_eq!(candidate.ops.len(), 3);
    }

    #[test]
    fn test_fusion_optimizer() {
        let mut optimizer = FusionOptimizer::new();

        optimizer.detect_fusions().expect("Should detect fusions");
        assert!(!optimizer.candidates().is_empty());

        let applied = optimizer.apply_fusions().expect("Should apply fusions");
        assert!(applied > 0);
        assert_eq!(optimizer.total_fusions(), applied);
    }

    #[test]
    fn test_fuse_elementwise() {
        let ops = vec!["add".to_string(), "mul".to_string(), "relu".to_string()];
        let fused = fuse_elementwise_ops(&ops).expect("Should fuse");

        assert_eq!(fused, "fused_add_mul_relu");
    }

    #[test]
    fn test_fuse_matmul_bias() {
        assert_eq!(
            fuse_matmul_bias(true, Some("relu")),
            "fused_matmul_bias_relu"
        );
        assert_eq!(fuse_matmul_bias(true, None), "fused_matmul_bias");
        assert_eq!(fuse_matmul_bias(false, Some("gelu")), "fused_matmul_gelu");
    }

    #[test]
    fn test_fuse_conv() {
        assert_eq!(
            fuse_conv_bn_activation(true, Some("relu")),
            "fused_conv_bn_relu"
        );
        assert_eq!(
            fuse_conv_bn_activation(false, Some("swish")),
            "fused_conv_swish"
        );
    }

    #[test]
    fn test_fusion_result() {
        let result = FusionResult::new(3, 1.5, 8192);

        assert!(result.has_changes());
        assert_eq!(result.fusions, 3);
        assert_eq!(result.speedup, 1.5);
    }
}
