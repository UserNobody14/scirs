# SciRS2 v0.2.0 CI/CD Infrastructure

This directory contains production-grade CI/CD workflows for the SciRS2 scientific computing workspace.

## 🚀 Workflow Overview

### Core CI/CD Workflows

#### `ci.yml` - Main CI Pipeline
**Triggers**: Push to main/feature branches, Pull Requests

**Features**:
- **Multi-Platform Testing**: Linux, macOS, Windows
- **Multi-Rust Version**: stable, beta, nightly
- **Feature Matrix Testing**:
  - Default features
  - All features enabled
  - No default features (minimal)
  - GPU-specific features (CUDA, OpenCL, Metal)
- **Cross-Compilation Validation**: 8+ target platforms
  - Linux: x86_64-musl, aarch64, armv7
  - macOS: x86_64, aarch64
  - Windows: x86_64
  - WASM: unknown, wasi
- **Feature Combination Testing**: Validates module interoperability
- **Minimal Build Verification**: Ensures no bloat
- **Integration Tests**: Cross-module compatibility
- **Documentation Validation**: Zero broken links

#### `quality-gates.yml` - Code Quality Enforcement
**Triggers**: Push to main branches, Pull Requests

**Quality Checks**:
- ✅ **Zero Clippy Warnings** (strict enforcement)
  - `-D warnings` (deny all warnings)
  - `-D clippy::unwrap_used` (enforce no unwrap policy)
  - `-D clippy::panic` (no panics)
- ✅ **Zero Compiler Warnings** (`RUSTFLAGS="-D warnings"`)
- ✅ **Code Formatting** (`cargo fmt --check`)
- ✅ **Security Audit** (`cargo-audit --deny warnings`)
- ✅ **Dependency Checks** (`cargo-deny`)
  - License compliance
  - Vulnerability scanning
  - **COOLJAPAN Pure Rust Policy enforcement**:
    - ❌ Blocks OpenBLAS/BLAS dependencies
    - ❌ Blocks bincode (use oxicode)
    - ⚠️ Warns about rustfft (prefer OxiFFT)
- ✅ **Unsafe Code Audit** (`cargo-geiger`)
- ✅ **License Compliance Check**
- ✅ **Code Complexity Analysis** (`tokei`)
- ✅ **Dead Code Detection** (`cargo-udeps`)
- ✅ **MSRV Check** (Rust 1.75.0+)

#### `coverage.yml` - Test Coverage Reporting
**Triggers**: Push to main branches, Pull Requests, Weekly schedule

**Coverage Methods**:
- **Tarpaulin**: Primary coverage tool (LLVM engine)
  - Uploads to Codecov
  - Generates HTML reports
  - Enforces minimum 70% coverage threshold
- **LLVM-COV**: Alternative coverage method
  - Cross-validation with Tarpaulin
- **Per-Crate Coverage**: Individual crate analysis
  - Detailed coverage for each module
  - Identifies low-coverage areas
- **PR Comments**: Automatic coverage reports on pull requests

#### `performance.yml` - Performance Regression Detection
**Triggers**: Push to main branches, Pull Requests, Twice daily (6 AM/PM UTC), Manual dispatch

**Performance Testing**:
- **Benchmark Matrix**: Ubuntu, macOS × (quick, comprehensive, stress)
- **Criterion Benchmarks**: Statistical performance analysis
- **v0.2.0 Validation Benchmarks**: Special validation suite
- **Baseline Comparison**: Automatic regression detection
  - Compares against master branch
  - 5% threshold for regressions/improvements
- **GPU Benchmarks**: CUDA performance testing (when available)
- **Memory Profiling**: Valgrind massif analysis
- **Performance Summary**: Aggregated results with visualization
- **PR Comments**: Automatic performance reports
- **Regression Alerts**: Fails CI on significant degradation

#### `nightly.yml` - Comprehensive Nightly Testing
**Triggers**: Daily at 2 AM UTC, Manual dispatch

**Extended Testing**:
- **Extended Test Suite**: All tests including ignored ones
- **Property-Based Testing**: 10,000 iterations with QuickCheck/Proptest
- **Stress Tests**: Memory-intensive workloads
- **Fuzz Testing**: `cargo-fuzz` on critical modules (5 min each)
- **Miri**: Undefined behavior detection
- **Cross-Compilation Matrix**: 9 target platforms
- **Documentation Validation**: Comprehensive doc checks
- **Security Audit**: Full security scan
- **Memory Leak Detection**: Valgrind full leak check
- **Dependency Update Check**: `cargo-outdated`
- **Platform Edge Cases**: 6 OS versions (Ubuntu, macOS, Windows)
- **Large Dataset Tests**: Performance with large data
- **Automatic Issue Creation**: On nightly failure

#### `release.yml` - Release Automation
**Triggers**: Git tags (`v*.*.*`), Manual dispatch

**Release Process**:
1. **Validate Release**:
   - Version consistency check across workspace
   - CHANGELOG validation
   - Comprehensive test suite
   - Documentation build verification
   - Dry-run publish test
2. **Build Artifacts**: Cross-platform binaries
   - Linux: x86_64, x86_64-musl
   - macOS: x86_64, aarch64 (Apple Silicon)
   - Windows: x86_64
3. **Documentation**: API docs generation and packaging
4. **Changelog Generation**: From git commit history
5. **GitHub Release**: Automatic release creation with artifacts
6. **Crates.io Publishing**: **Dry-run only** (per COOLJAPAN policy)
   - Dependency-ordered publishing
   - 30-second delay between crates
   - ⚠️ Always uses `--dry-run` flag
7. **Documentation Deployment**: GitHub Pages deployment

#### `pypi-publish.yml` - PyPI Python Package Publishing
**Triggers**: Manual workflow dispatch, Release tags (`v*`)

**Features**:
- Cross-platform wheel building (Linux x86_64/aarch64, macOS x86_64/arm64, Windows x64)
- Maturin-based build system for PyO3 bindings
- Trusted publishing with OIDC authentication
- TestPyPI and PyPI support

## 📋 Quality Standards

### Zero Warnings Policy
All code must compile with zero warnings:
- Compiler warnings: `RUSTFLAGS="-D warnings"`
- Clippy warnings: `-- -D warnings`
- Documentation warnings: `RUSTDOCFLAGS="-D warnings"`

### No Unwrap Policy
- Enforced via Clippy: `-D clippy::unwrap_used`
- Use `.expect()` with descriptive messages
- Proper error handling with `Result`/`Option`

### COOLJAPAN Pure Rust Policy
Enforced via `cargo-deny` and quality gates:
- ❌ **No OpenBLAS/BLAS**: Use OxiBLAS instead
- ❌ **No bincode**: Use oxicode instead
- ⚠️ **Prefer OxiFFT** over rustfft
- ✅ **100% Pure Rust** in default features
- Feature gates required for C/Fortran dependencies

## 🔧 Setup Requirements

### GitHub Secrets (Optional)
- `CODECOV_TOKEN`: For coverage reporting to Codecov
- `CARGO_REGISTRY_TOKEN`: For crates.io (not used due to dry-run policy)

### Repository Settings

#### Branch Protection (Recommended for `master`)
Required status checks:
- `Test Matrix` (ci.yml)
- `Clippy (Zero Warnings)` (quality-gates.yml)
- `Compiler Warnings (Zero Warnings)` (quality-gates.yml)
- `Code Formatting` (quality-gates.yml)
- `Security Audit` (quality-gates.yml)
- `Documentation` (ci.yml)

#### Actions Permissions
- Allow GitHub Actions to create and approve pull requests
- Allow GitHub Actions to write to repository

### Local Development Tools

```bash
# Essential CI/CD tools
cargo install cargo-nextest      # Fast test runner
cargo install cargo-audit        # Security auditing
cargo install cargo-deny         # Policy enforcement
cargo install cargo-tarpaulin    # Coverage (Linux only)
cargo install cargo-llvm-cov     # Coverage (cross-platform)

# Quality tools
cargo install cargo-geiger       # Unsafe code audit
cargo install cargo-outdated     # Dependency updates
cargo install cargo-udeps        # Unused dependencies

# Documentation
cargo install mdbook             # Book generation
cargo install cargo-readme       # README generation

# Optional development tools
cargo install cargo-watch        # File watching
cargo install cargo-expand       # Macro expansion
cargo install tokei              # Code statistics
```

## 📊 CI/CD Features

### Performance Regression Detection
- Automatic baseline comparison
- 5% threshold for regression/improvement classification
- Statistical analysis via Criterion
- Trend visualization
- PR comments with performance impact

### Security Scanning
- **cargo-audit**: Vulnerability database scanning
- **cargo-deny**: Dependency policy enforcement
- **cargo-geiger**: Unsafe code detection
- License compliance checking
- Pure Rust policy validation

### Coverage Reporting
- Multi-tool validation (Tarpaulin + LLVM-COV)
- Per-crate coverage analysis
- 70% minimum coverage threshold
- Codecov integration
- HTML reports for detailed analysis

### Cross-Platform Validation
- 3 primary platforms (Linux, macOS, Windows)
- 6 OS versions for edge case testing
- 8+ cross-compilation targets
- WASM support validation

## 🚨 Troubleshooting

### Common Issues

#### Quality Gates Failing

**Clippy Warnings**:
```bash
# Fix locally
cargo clippy --workspace --all-targets --all-features --fix
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

**Formatting**:
```bash
cargo fmt --all
```

**Unwrap Policy Violations**:
```bash
# Replace unwrap() with expect() or proper error handling
# BAD:  value.unwrap()
# GOOD: value.expect("descriptive message about what went wrong")
# BEST: value?  // or proper error handling
```

#### Performance Regression

Update baseline after intentional performance changes:
```bash
cargo bench --workspace -- --save-baseline new-baseline
```

#### Coverage Below Threshold

Add tests for uncovered code paths:
```bash
# Generate coverage report locally
cargo tarpaulin --workspace --all-features --out Html
# Open tarpaulin-report.html to see uncovered lines
```

### Local CI Reproduction

```bash
# Run the same checks as CI
cargo nextest run --workspace --all-features --no-fail-fast
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
cargo doc --workspace --all-features --no-deps
cargo audit
cargo deny check
```

## 📈 Monitoring and Artifacts

### Generated Artifacts
- **Coverage Reports**: HTML, LCOV, Cobertura XML
- **Performance Results**: Criterion reports, regression analysis
- **Security Reports**: Audit results, unsafe code analysis
- **Build Artifacts**: Cross-platform binaries
- **Documentation**: API docs, coverage summaries
- **Quality Reports**: Clippy, formatting, complexity analysis

### Retention Periods
- Coverage reports: 30 days
- Performance data: 90 days
- Security audits: 90 days
- Build artifacts: 90 days
- Nightly reports: 90 days

## 🎯 v0.2.0 Enhancements

### New Features
- ✅ Zero warnings enforcement (clippy + compiler)
- ✅ COOLJAPAN Pure Rust Policy automation
- ✅ Multi-tool coverage validation
- ✅ Performance regression detection with baselines
- ✅ GPU benchmark support (CUDA, OpenCL, Metal)
- ✅ Comprehensive cross-compilation matrix
- ✅ Automated release workflow
- ✅ Per-crate coverage analysis
- ✅ Memory leak detection (valgrind)
- ✅ Fuzz testing (cargo-fuzz)
- ✅ Undefined behavior detection (Miri)
- ✅ Platform edge case testing (6 OS versions)

### Quality Improvements
- Strict dependency policy enforcement
- License compliance automation
- Unsafe code auditing
- Dead code detection
- MSRV verification
- Documentation completeness validation

## 🔄 Workflow Execution Schedule

| Workflow | Frequency | Duration |
|----------|-----------|----------|
| CI | On push/PR | ~15-30 min |
| Quality Gates | On push/PR | ~10-15 min |
| Coverage | On push/PR + Weekly | ~20-30 min |
| Performance | On push/PR + 2x daily | ~30-60 min |
| Nightly | Daily 2 AM UTC | ~2-3 hours |
| Release | On git tags | ~45-60 min |

## 📚 Best Practices

### For Contributors
1. Run local checks before pushing: `cargo clippy && cargo fmt && cargo test`
2. Add tests for new features (maintain >70% coverage)
3. Update documentation for API changes
4. Follow the "No Unwrap" policy
5. Respect the Pure Rust Policy (no OpenBLAS/bincode)

### For Maintainers
1. Review quality gate reports on PRs
2. Monitor nightly test results
3. Update baselines after intentional performance changes
4. Keep dependencies up to date
5. Respond to security audit findings promptly

## 🐍 PyPI Publishing

See [PyPI Publishing Workflow](#pypi-publishyml---pypi-python-package-publishing) section above.

For detailed PyPI publishing instructions, refer to lines 239-288 of this README.

## 🔗 Related Documentation

- [Contributing Guidelines](../../CONTRIBUTING.md)
- [Security Policy](../../SECURITY.md)
- [Workspace README](../../README.md)
- [CHANGELOG](../../CHANGELOG.md)

---

**COOLJAPAN OU (Team Kitasan)** | SciRS2 v0.2.0 | Apache-2.0 License
