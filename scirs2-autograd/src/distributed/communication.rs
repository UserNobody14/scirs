//! Communication primitives for distributed training

use crate::{error::AutogradError, Float, NdArray, Result};

/// Communication operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommOp {
    /// Broadcast from one rank to all
    Broadcast,
    /// Reduce (sum) from all ranks to one
    Reduce,
    /// AllReduce - reduce and broadcast result
    AllReduce,
    /// Gather data from all ranks to one
    Gather,
    /// Scatter data from one rank to all
    Scatter,
    /// All-to-all communication
    AllToAll,
}

/// Communication handle for asynchronous operations
pub struct CommHandle {
    /// Operation type
    pub op: CommOp,
    /// Is operation complete
    completed: bool,
}

impl CommHandle {
    /// Create a new communication handle
    pub fn new(op: CommOp) -> Self {
        Self {
            op,
            completed: false,
        }
    }

    /// Wait for operation to complete
    pub fn wait(&mut self) -> Result<()> {
        // In real implementation, would wait for actual communication
        self.completed = true;
        Ok(())
    }

    /// Check if operation is complete
    pub fn is_complete(&self) -> bool {
        self.completed
    }
}

/// Compression strategy for gradient communication
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionStrategy {
    /// No compression
    None,
    /// Quantization to lower precision
    Quantize,
    /// Sparsification - send only large gradients
    Sparsify,
    /// Combination of quantization and sparsification
    Hybrid,
}

/// Compress gradients for efficient communication
pub fn compress_gradient<T: Float>(
    gradient: &NdArray<T>,
    strategy: CompressionStrategy,
) -> Result<Vec<u8>> {
    match strategy {
        CompressionStrategy::None => {
            // Just serialize as bytes
            let slice = gradient.as_slice().ok_or_else(|| {
                AutogradError::compute_error("Gradient is not contiguous".to_string())
            })?;

            let bytes: Vec<u8> = slice
                .iter()
                .flat_map(|&x| {
                    let f: f64 = x.to_f64().unwrap_or(0.0);
                    f.to_le_bytes().to_vec()
                })
                .collect();

            Ok(bytes)
        }
        _ => {
            // Other compression strategies would be implemented here
            Err(AutogradError::not_implemented(format!(
                "Compression strategy {:?} not implemented",
                strategy
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comm_handle() {
        let mut handle = CommHandle::new(CommOp::AllReduce);
        assert!(!handle.is_complete());

        handle.wait().expect("Should wait");
        assert!(handle.is_complete());
    }

    #[test]
    fn test_comm_op_equality() {
        assert_eq!(CommOp::Broadcast, CommOp::Broadcast);
        assert_ne!(CommOp::Broadcast, CommOp::Reduce);
    }
}
