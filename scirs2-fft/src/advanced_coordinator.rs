//! Advanced Mode Coordinator for FFT Operations
//!
//! This module provides an advanced AI-driven coordination system for FFT operations,
//! featuring intelligent algorithm selection, adaptive optimization, real-time performance
//! tuning, and cross-domain signal processing intelligence.
//!
//! # API Consistency
//!
//! This coordinator follows the standardized Advanced API patterns:
//! - Consistent naming: `enable_method_selection`, `enable_adaptive_optimization`
//! - Unified configuration fields across all Advanced coordinators
//! - Standard factory functions: `create_advanced_fft_coordinator()`
//!
//! # Features
//!
//! - **Intelligent Algorithm Selection**: AI-driven selection of optimal FFT algorithms
//! - **Adaptive Performance Tuning**: Real-time optimization based on signal characteristics
//! - **Multi-dimensional Coordination**: Unified optimization across 1D, 2D, and N-D FFTs
//! - **Memory-Aware Planning**: Intelligent memory management and caching strategies
//! - **Hardware-Adaptive Optimization**: Automatic tuning for different CPU/GPU architectures
//! - **Signal Pattern Recognition**: Advanced pattern analysis for optimization hints
//! - **Quantum-Inspired Optimization**: Next-generation optimization using quantum principles
//! - **Cross-Domain Knowledge Transfer**: Learning from diverse signal processing tasks

// Re-export the module implementation
#[allow(clippy::module_inception)]
mod advanced_coordinator;

// Re-export all public items
pub use advanced_coordinator::*;
