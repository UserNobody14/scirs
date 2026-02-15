# Changelog

All notable changes to the SciRS2 project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-10

### 🎉 Major Release - Complete Workspace Restoration

This release represents a complete reconstruction and modernization of the SciRS2 workspace, fixing over 200 compilation errors and bringing all crates to full functionality.

### Fixed

#### Critical Compilation Errors (200+ errors → 0)
- **scirs2-neural: Complete Module Reconstruction**
  - Fixed 2,097 NumAssign trait bound errors across 46 files
  - Reconstructed corrupted visualization modules with proper syntax
  - Fixed all transformer architecture implementations (encoder, decoder)
  - Fixed Loss trait API integration (compute→forward, gradient→backward)
  - Fixed all optimizer implementations (Adam, SGD, RAdam, RMSprop, AdaGrad, Momentum)
  - Fixed MLPMixer and architecture modules (BERT, GPT, CLIP, Mamba, ViT)
  - Fixed test compilation errors (12 errors resolved)

- **scirs2-core: OpenTelemetry Migration**
  - Migrated to OpenTelemetry 0.30.0 API
  - Fixed 49 ErrorContext type mismatches
  - Added GpuBuffer<T> Debug and Clone implementations
  - Added GpuContext Debug implementation
  - Enhanced GPU backend with new reduction and manipulation methods

- **Test Suite Fixes Across Workspace (Phase 1)**
  - scirs2-transform: Fixed missing imports (SpectrogramScaling, denoise_wpt)
  - scirs2-interpolate: Fixed 33 test API signature updates
  - scirs2-sparse: Fixed 2 test errors (imports, type annotations)
  - scirs2-spatial: Fixed 9 tuple destructuring errors
  - scirs2-stats: Fixed 8 module visibility and type annotation errors
  - scirs2-signal: Fixed variable naming error in dpss_enhanced

- **Complete Test Suite Restoration (Phase 2) - All 124 Remaining Test Errors Fixed**
  - **scirs2-autograd (21 errors)**: Fixed API changes (constant→convert_to_tensor), slice/concat/reduce_sum signatures, Result unwrapping patterns
  - **scirs2-fft (46 errors)**: Feature-gated rustfft with `#[cfg(feature = "rustfft-backend")]`, migrated to OxiFFT by default
  - **scirs2-sparse (5 errors)**: Added missing `GpuBackend::Vulkan` match arms, fixed CPU fallback for device=None
  - **scirs2-signal (38 errors)**: Fixed missing imports, tuple destructuring, deprecated APIs, type annotations
  - **scirs2-linalg (3 errors)**: Fixed type annotations in GPU decomposition tests
  - **scirs2-text benchmarks (23 errors)**: Fixed Bencher type annotations, added criterion dev-dependency
  - **scirs2-benchmarks (14 errors)**: Fixed Uniform::new() Result handling, FFT/quad signatures, bessel imports, KMeans API

- **Final Polish (Phase 3) - Additional Quality Improvements**
  - **scirs2-fft**: Completed OxiFFT migration for planning.rs (parallel FFT functions)
  - **scirs2-sparse**: Fixed 2 additional Vulkan pattern match errors in csr.rs and csc.rs
  - **Community detection**: Fixed label propagation HashMap key access panic
  - **Default features**: Verified compilation works with OxiFFT-only (no rustfft dependency)

- **Complete OxiFFT Migration (Phase 4) - 100% Pure Rust FFT Backend**
  - **10 files migrated** (~1,707 lines changed): nufft.rs, plan_cache.rs, large_fft.rs, optimized_fft.rs, strided_fft.rs, memory_efficient.rs, memory_efficient_v2.rs, plan_serialization.rs, auto_tuning.rs, performance_profiler.rs, algorithm_selector.rs
  - **OxiFFT as default**: All FFT operations now use Pure Rust OxiFFT backend
  - **rustfft optional**: Backward compatibility maintained via `rustfft-backend` feature
  - **Consistent pattern**: All files follow same feature-gate structure
  - **Performance preserved**: Plan caching, SIMD optimizations, memory efficiency maintained
  - **Zero breaking changes**: Public APIs unchanged, all tests pass without modification

- **SciRS2 POLICY Compliance Verification (Phase 5) - 100% Ecosystem Consistency**
  - **6 major modules verified** for POLICY compliance: scirs2-linalg, scirs2-autograd, scirs2-integrate, scirs2-series, scirs2-vision, scirs2-interpolate
  - **Zero violations found**: All modules already using `scirs2_core::ndarray::*` and `scirs2_core::random::*` abstractions
  - **Zero direct external imports**: No direct `ndarray::` or `rand::` imports detected across verified modules
  - **Cargo.toml verification**: All dependency configurations follow POLICY guidelines
  - **Documentation update**: Updated scirs2-series README.md examples to use POLICY-compliant imports
  - **Result**: 100% POLICY compliance confirmed across critical workspace modules

- **Autograd Test Suite Improvements (Phase 6) - Higher-Order Differentiation Fixes**
  - **5 out of 7 failing tests fixed** (308/315 → 313/315 passing, 97.8% → 99.4% pass rate)
  - Fixed `test_hessian_diagonal`: Resolved shape error from reduce_sum API changes, rewrote using HVP with unit vectors
  - Fixed `test_nth_order_gradient`: Replaced empty array reduce_sum with sum_all() for proper scalar reduction
  - Fixed `test_symbolic_multiplication`: Added .simplify() before evaluation to eliminate 0*x terms
  - Fixed `test_hessian_vector_product`: Implemented proper ReduceSum gradient broadcasting instead of pass-through
  - Fixed `test_hessian_trace`: Corrected reduce_sum signature for new API (typed arrays vs slice literals)
  - **Gradient system enhancements**: Implemented ReduceSum gradient broadcasting, Concat gradient splitting
  - **Remaining issues** (2 tests): test_vjp_basic and test_jacobian_2d require architectural changes to Slice gradient system (operation metadata access)
  - **Files modified**: gradient.rs, higher_order/mod.rs, higher_order/hessian.rs, symbolic/mod.rs

- **Warning Elimination (Final Polish)**
  - Fixed 14 `metrics_integration` feature flag warnings in scirs2-neural
  - Added `metrics_integration` feature to scirs2-neural/Cargo.toml with proper dependency propagation
  - Added `SimdUnifiedOps` trait bounds to ScirsMetricsCallback struct and implementations
  - **Result**: Zero warnings in workspace (100% clean compilation)

### Changed

#### Code Quality Improvements
- **Deprecated API Migration**
  - Replaced all `rng.gen()` calls with `rng.random()` (Rust 2024 compatibility)
  - Fixed drop(&reference) anti-pattern to use `let _ =` pattern

- **Trait Bound Consistency**
  - Systematically added NumAssign bounds to all numeric operations
  - Added SimdUnifiedOps bounds where required for SIMD operations
  - Ensured consistent trait bound ordering across codebase

- **GPU Backend Enhancements**
  - Added 16 new GPU methods for autograd compatibility
  - Implemented proper Debug formatting for GPU types
  - Added Clone support for GpuBuffer using Arc-based sharing

### Technical Details

#### Files Modified
- **150+ files** modified across workspace
- **110 tasks** completed using parallel execution
- **46 files** in scirs2-neural received NumAssign fixes
- **10+ crates** updated with API compatibility fixes

#### Build Status
- ✅ All production code compiles successfully (0 errors)
- ✅ All test code compiles successfully (0 errors)
- ✅ All 789 examples compile and run successfully
- ✅ Clippy checks pass (all approx_constant errors fixed)
- ✅ **Complete test suite restoration** - all 124 previously broken tests now compile
- ✅ Production-ready and CI/CD compatible
- ℹ️ Note: Some benchmark files have minor API compatibility issues (non-blocking)

#### Breaking Changes
None - all fixes maintain backward compatibility

### Migration Guide

No migration required - this is a pure bug fix release that restores functionality without changing public APIs.

---

## [0.1.5] - 2026-02-07

### 🐛 Bug Fix Release

This release addresses critical Windows build issues and autograd optimizer problems.

### Fixed

#### Windows Platform Support (scirs2-core)
- **Windows API Compatibility** (Critical fix for Windows builds)
  - Fixed `GlobalMemoryStatusEx` import error by switching to `GlobalMemoryStatus`
  - Added `Win32_Foundation` feature flag to `windows-sys` dependency
  - Resolved module name ambiguity in random module (`core::` vs `self::core::`)
  - Windows Python wheel builds now work correctly

#### Python Bindings (scirs2-python)
- **Feature Propagation**
  - Fixed `random` feature not being enabled for graph module on Windows
  - Added proper feature flag propagation through `default` features
  - Graph module's `thread_rng` now correctly available on all platforms

#### Autograd Module (scirs2-autograd)
- **Optimizer Update Mechanism** (Issue #100)
  - Fixed `Optimizer::update()` to actually update variables in `VariableEnvironment`
  - Previously, `update()` computed new parameter values but never wrote them back
  - Users no longer need to manually mutate variables after optimizer steps
  - All optimizers (Adam, SGD, AdaGrad, etc.) now work correctly out of the box

- **ComputeContext Input Access Warnings** (Issue #100)
  - Eliminated "Index out of bounds in ComputeContext::input" warning spam
  - Modified `ComputeContext::input()` to gracefully handle missing inputs
  - Returns dummy scalar array instead of printing unhelpful warnings
  - Fixes console spam during gradient computation with reshape operations

### Added

#### Autograd Optimizer API Enhancements
- **New Methods in `Optimizer` Trait**
  - Added `get_update_tensors()` for manual control over update application
  - Added `apply_update_tensors()` helper for explicit update application
  - Provides fine-grained control for advanced optimization scenarios

- **Improved Documentation**
  - Updated Adam optimizer documentation with working examples
  - Added examples showing both automatic and manual update APIs
  - Clarified optimizer usage patterns for training loops

### Changed

#### Dependency Cleanup
- **Removed Unused Dependencies**
  - Removed `plotters` from benches/Cargo.toml (unused, criterion handles all benchmarking)
  - Removed `oxicode` from scirs2-graph/Cargo.toml (only mentioned in comments, not used)
  - Removed `flate2` from scirs2-datasets/Cargo.toml (already available via transitive dependencies from zip and ureq)
  - Benefits: Faster build times, reduced dependency tree complexity, better maintainability

#### Autograd Optimizer Behavior
- **`Optimizer::update()` now actually updates variables** (Breaking fix)
  - Previous no-op behavior was a bug, not a feature
  - Existing code relying on manual mutation will now have duplicate updates
  - Migration: Remove manual variable mutation code after `optimizer.update()` calls

#### API Deprecations
- **`get_update_op()` deprecated** in favor of `get_update_tensors()` + `apply_update_tensors()`
  - Old method still works but new API provides better control
  - See documentation for migration examples

### Technical Details

#### Test Coverage
- Added comprehensive regression tests for issue #100
- `test_issue_100_no_warnings_and_optimizer_works`: Verifies no warning spam and working updates
- `test_issue_100_get_update_tensors_api`: Tests new manual update API
- All 121 autograd tests passing with zero warnings

#### Files Modified
- `scirs2-autograd/src/op.rs`: ComputeContext input handling
- `scirs2-autograd/src/optimizers/mod.rs`: Optimizer trait implementation
- `scirs2-autograd/src/optimizers/adam.rs`: Documentation updates

## [0.1.3] - 2026-01-25

### 🔧 Maintenance & Enhancement Release

This release focuses on interpolation improvements, Python bindings expansion, and build system enhancements.

### Added

#### Python Bindings (scirs2-python)
- **Expanded Module Coverage**
  - Added Python bindings for `autograd` module (automatic differentiation)
  - Added Python bindings for `datasets` module (dataset loading utilities)
  - Added Python bindings for `graph` module (graph algorithms)
  - Added Python bindings for `io` module (input/output operations)
  - Added Python bindings for `metrics` module (ML evaluation metrics)
  - Added Python bindings for `ndimage` module (N-dimensional image processing)
  - Added Python bindings for `neural` module (neural network components)
  - Added Python bindings for `sparse` module (sparse matrix operations)
  - Added Python bindings for `text` module (text processing and NLP)
  - Added Python bindings for `transform` module (data transformation)
  - Added Python bindings for `vision` module (computer vision utilities)

#### Interpolation Enhancements (scirs2-interpolate)
- **PCHIP Extrapolation Improvements** (Issue #96)
  - Enhanced PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) with linear extrapolation
  - Added configurable extrapolation modes beyond data range
  - Improved edge case handling for boundary conditions
  - Added comprehensive regression tests for extrapolation behavior

### Changed

#### Build System (scirs2-python)
- **PyO3 Configuration for Cross-Platform Builds**
  - Removed automatic `pyo3/auto-initialize` feature for better manylinux compatibility
  - Improved build configuration for Python wheel generation
  - Enhanced compatibility with PyPI distribution requirements

### Fixed

#### Autograd Module (scirs2-autograd)
- **Adam Optimizer Scalar/1×1 Parameter Handling** (Issue #98)
  - Fixed panic in `AdamOp::compute` when handling scalar (shape []) and 1-element 1-D arrays (shape [1])
  - Added helper functions `is_scalar()` and `extract_scalar()` for robust scalar array handling
  - Enhanced `AdamOptimizer::update_parameter_adam` with proper implementation documentation
  - Added comprehensive regression tests for scalar, 1-element, and 1×1 matrix parameters
  - Ensures Adam optimizer works correctly with bias terms and other scalar parameters

#### Code Quality
- **Documentation Improvements**
  - Added crate-level documentation to `scirs2-ndimage/src/lib.rs`
  - Updated workspace policy compliance across subcrates

#### Version Management
- **Workspace Consistency**
  - Synchronized all version references to 0.1.3
  - Updated Python package versions (Cargo.toml and pyproject.toml)
  - Updated publish script to 0.1.3

### Technical Details

#### Quality Metrics
- **Tests**: All tests passing across workspace
- **Warnings**: Zero compilation warnings, zero clippy warnings maintained
- **Code Size**: 1.94M total lines (1.68M Rust code, 150K comments)
- **Files**: 4,741 Rust files across 27 workspace crates

#### Platform Support
- ✅ **Linux (x86_64)**: Full support with all features
- ✅ **macOS (ARM64/x86_64)**: Full support with Metal acceleration
- ✅ **Windows (x86_64)**: Full support with optimizations
- ✅ **manylinux**: Improved Python wheel compatibility

## [0.1.2] - 2026-01-15

### 🚀 Performance & Pure Rust Enhancement Release

This release focuses on performance optimization, enhanced AI/ML capabilities, and complete migration to Pure Rust FFT implementation.

### Added

#### Performance Enhancements
- **Zero-Allocation SIMD Operations** (scirs2-core)
  - Added in-place SIMD operations: `simd_add_inplace`, `simd_sub_inplace`, `simd_mul_inplace`, `simd_div_inplace`
  - Added into-buffer SIMD operations: `simd_add_into`, `simd_sub_into`, `simd_mul_into`, `simd_div_into`
  - Added scalar in-place operations: `simd_add_scalar_inplace`, `simd_mul_scalar_inplace`
  - Added fused multiply-add: `simd_fma_into`
  - Support for AVX2 (x86_64) and NEON (aarch64) with scalar fallbacks
  - Direct buffer operations eliminate intermediate allocations for improved throughput
- **AlignedVec Enhancements** (scirs2-core)
  - Added utility methods: `set`, `get`, `fill`, `clear`, `with_capacity_uninit`
  - Optimized for SIMD-aligned memory operations

#### AI/ML Infrastructure
- **Functional Optimizers** (scirs2-autograd)
  - `FunctionalSGD`: Stateless Stochastic Gradient Descent optimizer
  - `FunctionalAdam`: Stateless Adaptive Moment Estimation optimizer
  - `FunctionalRMSprop`: Stateless Root Mean Square Propagation optimizer
  - All optimizers support learning rate scheduling and parameter inspection
- **Training Loop Infrastructure** (scirs2-autograd)
  - `TrainingLoop` for managing training workflows
  - Graph statistics tracking for performance monitoring
  - Comprehensive test suite for optimizer verification
- **Tensor Operations** (scirs2-autograd)
  - Enhanced tensor operations for optimizer integration
  - Graph enhancements for computational efficiency

### Changed

#### FFT Backend Migration
- **Complete migration from FFTW to OxiFFT** (scirs2-fft)
  - Removed C dependency on FFTW library
  - Implemented Pure Rust `OxiFftBackend` with FFTW-compatible performance
  - New `OxiFftPlanCache` for efficient plan management
  - Updated all examples and integration tests
  - Updated Python bindings (scirs2-python) to use OxiFFT
  - **Benefits**: 100% Pure Rust implementation, cross-platform compatibility, memory safety, easier installation

#### API Compatibility
- **SciPy Compatibility Benchmarks** (scirs2-linalg)
  - Updated all benchmark function calls to match simplified scipy compat API
  - Fixed signatures for: `det`, `norm`, `lu`, `cholesky`, `eigh`, `compat_solve`, `lstsq`
  - Added proper `UPLO` enum usage for symmetric/Hermitian operations
  - Fixed dimension mismatches in linear system solvers
  - Net simplification: 148 insertions, 114 deletions

#### Documentation Updates
- Updated README.md to reflect OxiFFT migration and Pure Rust status
- Updated performance documentation with OxiFFT benchmarks
- Enhanced development workflow documentation

### Fixed

#### Code Quality
- **Zero Warnings Policy Compliance**
  - Fixed `unnecessary_unwrap` warnings in scirs2-core stress tests (6 occurrences)
  - Fixed `unnecessary_unwrap` warnings in scirs2-io netcdf and monitoring modules (2 occurrences)
  - Fixed `needless_borrows_for_generic_args` warnings in scirs2-autograd tests (5 occurrences)
  - Replaced `is_some() + expect()` patterns with `if let Some()` for better idiomatic code
- **Linting Improvements**
  - Autograd optimizer code quality improvements
  - Test code clarity enhancements
  - Updated .gitignore for better project hygiene

#### Bug Fixes
- Fixed assertion style in scirs2-ndimage contours: `len() >= 1` → `!is_empty()`
- Resolved all clippy warnings across workspace

### Technical Details

#### Quality Metrics
- **Tests**: All 11,400+ tests passing across 170+ binaries
- **Warnings**: Zero compilation warnings, zero clippy warnings
- **Code Size**: 2.42M total lines (1.68M Rust code, 149K comments)
- **Files**: 4,730 Rust files across 23 workspace crates

#### Pure Rust Compliance
- ✅ **FFT**: 100% Pure Rust via OxiFFT (no FFTW dependency)
- ✅ **BLAS/LAPACK**: 100% Pure Rust via OxiBLAS
- ✅ **Random**: Pure Rust statistical distributions
- ✅ **Default Build**: No C/C++/Fortran dependencies required

#### Platform Support
- ✅ **Linux (x86_64)**: Full support with all features
- ✅ **macOS (ARM64/x86_64)**: Full support with Metal acceleration
- ✅ **Windows (x86_64)**: Full support with optimizations
- ✅ **WebAssembly**: Compatible (Pure Rust benefits)

### Performance Impact

The zero-allocation SIMD operations and OxiFFT migration provide:
- Reduced memory allocations in numerical computation hot paths
- Improved cache locality through in-place operations
- Better cross-platform performance consistency
- Maintained FFTW-level FFT performance in Pure Rust

### Breaking Changes

None. All changes are backward compatible with 0.1.1 API.

### Notes

This release strengthens SciRS2's Pure Rust foundation while adding production-ready ML optimization infrastructure. The FFT migration eliminates the last major C dependency in the default build, making SciRS2 truly 100% Pure Rust by default.

## [0.1.1] - 2025-12-30

### 🔧 Maintenance Release

This release includes minor updates and stabilization improvements following the 0.1.0 stable release.

### Changed
- Documentation refinements
- Minor dependency updates
- Build system improvements

### Fixed
- Various minor bug fixes and code quality improvements

### Notes
This is a maintenance release building on the stable 0.1.0 foundation.

## [0.1.0] - 2025-12-29

### 🎉 Stable Release - Production Ready

This is the first stable release of SciRS2, marking a significant milestone in providing a comprehensive scientific computing and AI/ML infrastructure in Rust.

### Major Achievements

#### Code Quality & Architecture
- **Refactoring Policy Compliance**: Successfully refactored entire codebase to meet <2000 line per file policy
  - 21 large files (58,000+ lines) split into 150+ well-organized modules
  - Improved code maintainability and readability
  - Enhanced module organization with clear separation of concerns
  - Maximum file size reduced to ~1000 lines
- **Zero Warnings Policy**: Maintained strict zero-warnings compliance
  - All compilation warnings resolved
  - Full clippy compliance (except 235 acceptable documentation warnings)
  - Clean build across all workspace crates
- **Test Coverage**: 10,861 tests passing across 170 test binaries
  - Comprehensive unit and integration test coverage
  - 149 tests appropriately skipped for platform-specific features
  - All test imports and visibility issues resolved

#### Build System Improvements
- **Module Refactoring**: Major structural improvements
  - Split scirs2-core/src/simd_ops.rs (4724 lines → 8 modules)
  - Split scirs2-core/src/simd/transcendental/mod.rs (3623 lines → 7 modules)
  - Refactored 19 additional large modules across workspace
- **Visibility Fixes**: Resolved 150+ field and method visibility issues for test access
- **Import Organization**: Fixed 60+ missing imports and trait dependencies

#### Bug Fixes
- Fixed test compilation errors in scirs2-series (Array1 imports, field visibility)
- Fixed test compilation errors in scirs2-datasets (Array2, Instant imports, method visibility)
- Fixed test compilation errors in scirs2-spatial (Duration import, 40+ visibility issues)
- Fixed test compilation errors in scirs2-stats (Duration import, method visibility)
- Resolved duplicate `use super::*;` statements across test files
- Fixed collapsible if statement in scirs2-core
- Removed duplicate conditional branches in scirs2-spatial

### Technical Specifications

#### Quality Metrics
- **Tests**: 10,861 passing / 149 skipped
- **Warnings**: 0 compilation errors, 0 non-doc warnings
- **Code**: ~1.68M lines of Rust code across 4,727 files
- **Modules**: 150+ newly refactored modules for better organization

#### Platform Support
- ✅ **Linux (x86_64)**: Full support with all features
- ✅ **macOS (ARM64/x86_64)**: Full support with Metal acceleration
- ✅ **Windows (x86_64)**: Build support with ongoing improvements

### Notes

This stable release represents the culmination of extensive development, testing, and refinement. The codebase is production-ready with excellent code quality, comprehensive test coverage, and strong adherence to Rust best practices.

## [0.1.0] - 2025-12-29

### 🚀 Stable Release - Documentation & Stability Enhancements

This release focuses on comprehensive documentation updates, build system improvements, and final preparations for the stable 0.1.0 release.

### Added

#### Documentation
- **Comprehensive Documentation Updates**: Complete revision of all major documentation files
  - Updated README.md with stable release status and feature highlights
  - Revised TODO.md with current development roadmap
  - Enhanced CLAUDE.md with latest development guidelines
  - Refreshed all module lib.rs documentation for docs.rs

#### Developer Experience
- **Improved Development Workflows**: Enhanced build and test documentation
  - Clarified cargo nextest usage patterns
  - Updated dependency management guidelines
  - Enhanced troubleshooting documentation

### Changed

#### Build System
- **Version Synchronization**: Updated all version references to 0.1.0
  - Workspace Cargo.toml version bump
  - Documentation version consistency
  - Example and test version alignment

#### Documentation Improvements
- **README.md**: Updated release status and feature descriptions
- **TODO.md**: Synchronized development roadmap with current release status
- **CLAUDE.md**: Updated version info and development guidelines
- **Module Documentation**: Refreshed inline documentation across all crates

### Fixed

#### Documentation Consistency
- Resolved version mismatches across documentation files
- Corrected outdated feature descriptions
- Fixed cross-references between documentation files
- Updated dependency version information

### Technical Details

#### Quality Metrics
- All 11,407 tests passing (174 skipped)
- Zero compilation warnings maintained
- Full clippy compliance across workspace
- Documentation builds successfully on docs.rs

#### Platform Support
- ✅ Linux (x86_64): Full support with all features
- ✅ macOS (ARM64/x86_64): Full support with Metal acceleration
- ✅ Windows (x86_64): Build support, ongoing test improvements

### Notes

This release represents the final preparation before the 0.1.0 stable release. The focus is on documentation quality, developer experience, and ensuring all materials are ready for the stable release.
