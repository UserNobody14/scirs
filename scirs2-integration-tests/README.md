# SciRS2 Integration Tests

This package contains comprehensive cross-crate integration tests for SciRS2 v0.2.0.

## Overview

The integration tests verify that different SciRS2 modules work correctly together,
testing data flow, API compatibility, and end-to-end workflows.

## Test Coverage

### Cross-Module Workflows
- **Neural + Optimize**: ML training pipelines with optimization algorithms
- **FFT + Signal**: Spectral analysis and signal processing
- **Sparse + Linalg**: Sparse linear algebra operations
- **NDImage + Vision**: Image processing and computer vision workflows
- **Stats + Datasets**: Statistical data analysis

### Performance Tests
- End-to-end pipeline benchmarks
- Memory usage profiling
- GPU/CPU handoff efficiency
- Zero-copy transfer validation

## Running Tests

### All integration tests
```bash
cargo test --package scirs2-integration-tests
```

### Specific test file
```bash
# Neural + Optimize tests
cargo test --package scirs2-integration-tests --test integration neural_optimize

# FFT + Signal tests
cargo test --package scirs2-integration-tests --test integration fft_signal

# Performance tests
cargo test --package scirs2-integration-tests --test integration performance
```

### Property-based tests only
```bash
cargo test --package scirs2-integration-tests prop_
```

### Performance benchmarks (ignored by default)
```bash
cargo test --package scirs2-integration-tests --ignored
```

### With verbose output
```bash
cargo test --package scirs2-integration-tests -- --nocapture
```

## Test Structure

See `tests/README.md` for detailed information about:
- Test organization
- Available test fixtures
- Test utilities
- Property-based testing patterns
- Performance targets

## Features

Optional features can be enabled for specific tests:
```bash
# Run tests with CUDA support
cargo test --package scirs2-integration-tests --features cuda

# Run tests with SIMD optimizations
cargo test --package scirs2-integration-tests --features simd
```

## Development

### Adding New Tests

1. Choose the appropriate test file based on modules involved
2. Follow the no-unwrap policy (use `expect()` or proper error handling)
3. Use property-based testing where applicable
4. Add performance targets for critical paths
5. Document test objectives

### Test Utilities

Common utilities are available in `tests/integration/common/`:
- `test_utils.rs`: Helper functions for array creation, timing, memory checks
- Property test generators for dimensions, tolerances, etc.

### Test Fixtures

Standard test data is available in `tests/integration/fixtures/`:
- XOR dataset
- Linear regression data
- Sinusoidal signals
- Sparse matrices
- Test images
- Statistical distributions

## Current Status

The test infrastructure is complete with:
- ✅ Test organization and module structure
- ✅ Common utilities and helpers
- ✅ Test fixtures and data generators
- ✅ Property-based testing framework
- ✅ Performance measurement utilities

Many test implementations contain `TODO` markers indicating where actual
test logic should be added once module APIs stabilize. The structure and
patterns are ready for implementation.

## License

Apache-2.0

## Authors

COOLJAPAN OU (Team KitaSan)
