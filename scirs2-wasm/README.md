# SciRS2-WASM: Scientific Computing in WebAssembly

High-performance scientific computing library for JavaScript and TypeScript, powered by Rust and WebAssembly.

## Features

- **Pure Rust**: 100% safe Rust code compiled to WASM
- **High Performance**: Near-native performance in the browser
- **SIMD Support**: Optional WASM SIMD acceleration (wasm32-simd128)
- **TypeScript Support**: Full TypeScript type definitions
- **Zero Dependencies**: No external JavaScript dependencies
- **Memory Efficient**: Optimized for browser memory constraints
- **Async Operations**: Non-blocking computations

## Installation

### NPM

```bash
npm install scirs2-wasm
```

### Yarn

```bash
yarn add scirs2-wasm
```

### PNPM

```bash
pnpm add scirs2-wasm
```

## Quick Start

### Browser (ES Modules)

```javascript
import init, * as scirs2 from 'scirs2-wasm';

async function main() {
  // Initialize the WASM module
  await init();

  // Create arrays
  const a = new scirs2.WasmArray([1, 2, 3, 4]);
  const b = new scirs2.WasmArray([5, 6, 7, 8]);

  // Perform operations
  const sum = scirs2.add(a, b);
  const mean_a = scirs2.mean(a);
  const std_a = scirs2.std(a);

  console.log('Sum:', sum.to_array());
  console.log('Mean:', mean_a);
  console.log('Std:', std_a);
}

main();
```

### Node.js

```javascript
const scirs2 = require('scirs2-wasm');

async function main() {
  // WASM module is auto-initialized in Node.js

  // Create a 2D array (matrix)
  const matrix = scirs2.WasmArray.from_shape(
    [2, 2],
    [1, 2, 3, 4]
  );

  // Linear algebra operations
  const det = scirs2.det(matrix);
  const trace = scirs2.trace(matrix);

  console.log('Determinant:', det);
  console.log('Trace:', trace);
}

main();
```

### TypeScript

```typescript
import init, * as scirs2 from 'scirs2-wasm';

async function main(): Promise<void> {
  await init();

  // Type-safe array operations
  const arr: scirs2.WasmArray = scirs2.WasmArray.linspace(0, 10, 100);

  const mean: number = scirs2.mean(arr);
  const std: number = scirs2.std(arr);

  console.log(`Mean: ${mean}, Std: ${std}`);
}

main();
```

## API Reference

### Array Creation

```javascript
// From JavaScript array
const arr = new scirs2.WasmArray([1, 2, 3, 4]);

// From typed array
const typed = new Float64Array([1, 2, 3, 4]);
const arr2 = new scirs2.WasmArray(typed);

// With specific shape (2D matrix)
const matrix = scirs2.WasmArray.from_shape([2, 2], [1, 2, 3, 4]);

// Special arrays
const zeros = scirs2.WasmArray.zeros([3, 3]);
const ones = scirs2.WasmArray.ones([5]);
const full = scirs2.WasmArray.full([2, 3], 7.0);

// Range arrays
const linspace = scirs2.WasmArray.linspace(0, 1, 50);  // [0, 0.02, 0.04, ..., 1]
const arange = scirs2.WasmArray.arange(0, 10, 0.5);    // [0, 0.5, 1, ..., 9.5]
```

### Array Operations

```javascript
const a = new scirs2.WasmArray([1, 2, 3]);
const b = new scirs2.WasmArray([4, 5, 6]);

// Element-wise operations
const sum = scirs2.add(a, b);        // [5, 7, 9]
const diff = scirs2.subtract(a, b);  // [-3, -3, -3]
const prod = scirs2.multiply(a, b);  // [4, 10, 18]
const quot = scirs2.divide(a, b);    // [0.25, 0.4, 0.5]

// Dot product and matrix operations
const dot = scirs2.dot(a, b);        // 32
const matrix_a = scirs2.WasmArray.from_shape([2, 2], [1, 2, 3, 4]);
const matrix_b = scirs2.WasmArray.from_shape([2, 2], [5, 6, 7, 8]);
const matmul = scirs2.dot(matrix_a, matrix_b);

// Reductions
const total = scirs2.sum(a);    // 6
const average = scirs2.mean(a); // 2
const minimum = scirs2.min(a);  // 1
const maximum = scirs2.max(a);  // 3
```

### Statistical Functions

```javascript
const data = new scirs2.WasmArray([1, 2, 3, 4, 5]);

// Descriptive statistics
const mean = scirs2.mean(data);           // 3
const std = scirs2.std(data);             // ~1.41
const variance = scirs2.variance(data);   // ~2
const median = scirs2.median(data);       // 3

// Percentiles
const p25 = scirs2.percentile(data, 25);  // 2
const p75 = scirs2.percentile(data, 75);  // 4

// Correlation
const x = new scirs2.WasmArray([1, 2, 3, 4, 5]);
const y = new scirs2.WasmArray([2, 4, 6, 8, 10]);
const corr = scirs2.corrcoef(x, y);  // 1.0 (perfect correlation)

// Cumulative operations
const cumsum = scirs2.cumsum(data);   // [1, 3, 6, 10, 15]
const cumprod = scirs2.cumprod(data); // [1, 2, 6, 24, 120]
```

### Linear Algebra

```javascript
const matrix = scirs2.WasmArray.from_shape([3, 3], [
  1, 2, 3,
  0, 1, 4,
  5, 6, 0
]);

// Matrix properties
const det = scirs2.det(matrix);           // Determinant
const trace = scirs2.trace(matrix);       // Sum of diagonal
const rank = scirs2.rank(matrix);         // Matrix rank
const norm = scirs2.norm_frobenius(matrix); // Frobenius norm

// Matrix operations
const inv = scirs2.inv(matrix);           // Matrix inverse
const transpose = matrix.transpose();     // Matrix transpose

// Solve linear system Ax = b
const A = scirs2.WasmArray.from_shape([2, 2], [3, 1, 1, 2]);
const b = new scirs2.WasmArray([9, 8]);
const x = scirs2.solve(A, b);  // Solution: x = [2, 3]
```

### Random Number Generation

```javascript
// Uniform random [0, 1)
const uniform = scirs2.random_uniform([100]);

// Normal distribution (mean=0, std=1)
const normal = scirs2.random_normal([1000], 0, 1);

// Random integers [0, 10)
const integers = scirs2.random_integers([50], 0, 10);

// Exponential distribution (lambda=1.5)
const exponential = scirs2.random_exponential([100], 1.5);
```

### Performance Monitoring

```javascript
// Time operations
const timer = new scirs2.PerformanceTimer('Matrix multiplication');

const a = scirs2.WasmArray.from_shape([1000, 1000], /* data */);
const b = scirs2.WasmArray.from_shape([1000, 1000], /* data */);
const result = scirs2.dot(a, b);

timer.log_elapsed();  // Logs: "Matrix multiplication: 45.123ms"

// Check capabilities
const caps = scirs2.capabilities();
console.log('SIMD support:', caps.simd);
console.log('Available features:', caps.features);
```

## Building from Source

### Prerequisites

- Rust 1.70+ with `wasm32-unknown-unknown` target
- wasm-pack
- Node.js 18+

### Build Steps

```bash
# Install Rust target
rustup target add wasm32-unknown-unknown

# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for bundlers (webpack, rollup, etc.)
npm run build

# Build for web (ES modules)
npm run build:web

# Build for Node.js
npm run build:nodejs

# Build with SIMD support
npm run build:simd

# Optimize with wasm-opt
npm run optimize
```

### Testing

```bash
# Run tests in headless Firefox
npm test

# Run tests in headless Chrome
npm run test:chrome

# Run tests in Node.js
npm run test:node
```

## Performance

SciRS2-WASM provides near-native performance for scientific computing tasks:

- **Array operations**: 80-95% of native performance
- **Matrix multiplication**: Up to 90% with SIMD
- **Statistical functions**: 85-95% of native
- **Random number generation**: 70-85% (limited by WASM RNG)

### SIMD Acceleration

For maximum performance, build with SIMD support:

```bash
npm run build:simd
```

Note: SIMD requires browser support for wasm-simd128 feature.

## Browser Compatibility

- Chrome 91+ (with SIMD: 91+)
- Firefox 89+ (with SIMD: 89+)
- Safari 16.4+ (with SIMD: limited)
- Edge 91+ (with SIMD: 91+)
- Node.js 18+ (with SIMD: 20+)

## Memory Management

WASM has a linear memory model. Best practices:

1. **Reuse arrays** when possible to avoid allocations
2. **Process in chunks** for large datasets
3. **Monitor memory** usage with browser DevTools
4. **Dispose arrays** explicitly if needed (future feature)

## Examples

See the `examples/` directory for complete examples:

- `basic-arrays.html` - Array creation and manipulation
- `statistics.html` - Statistical analysis
- `linear-algebra.html` - Matrix operations
- `performance.html` - Performance benchmarking
- `node-example.js` - Node.js usage

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

Apache-2.0

## Authors

COOLJAPAN OU (Team KitaSan)

## Acknowledgments

- Built with [wasm-bindgen](https://github.com/rustwasm/wasm-bindgen)
- Powered by [ndarray](https://github.com/rust-ndarray/ndarray)
- Part of the [SciRS2](https://github.com/cool-japan/scirs) ecosystem
