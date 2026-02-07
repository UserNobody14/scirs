# scirs2-numpy

NumPy integration for SciRS2 scientific computing library.

This crate provides seamless interoperability between SciRS2's ndarray-based data structures and Python's NumPy arrays through PyO3.

## Features

- Zero-copy data sharing between Rust and Python when possible
- Support for common numeric types (f32, f64, i32, i64, etc.)
- Complex number support
- Multi-dimensional array support

## Usage

This crate is primarily used internally by the scirs2-python bindings. For direct usage, see the scirs2-python package.

## License

Licensed under the Apache License 2.0. See LICENSE for details.
