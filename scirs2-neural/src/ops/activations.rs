//! Activation functions for neural networks
//!
//! This module provides standalone activation functions that operate on ndarray Arrays.
//! These functions are designed to work seamlessly with both direct ndarray operations
//! and scirs2-autograd Tensors.

use crate::error::Result;
use scirs2_core::ndarray::{Array, IxDyn, Zip};
use scirs2_core::numeric::Float;
use std::fmt::Debug;

/// ReLU (Rectified Linear Unit) activation function
///
/// Applies the element-wise function: `f(x) = max(0, x)`
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// Array with ReLU applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::relu;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
/// let output = relu(&input).expect("ReLU failed");
/// ```
pub fn relu<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let mut output = input.clone();
    let zero = F::zero();
    Zip::from(&mut output).for_each(|x| {
        if *x < zero {
            *x = zero;
        }
    });
    Ok(output)
}

/// Sigmoid activation function
///
/// Applies the element-wise function: `f(x) = 1 / (1 + exp(-x))`
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// Array with sigmoid applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::sigmoid;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
/// let output = sigmoid(&input).expect("Sigmoid failed");
/// ```
pub fn sigmoid<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let mut output = input.clone();
    let one = F::one();
    Zip::from(&mut output).for_each(|x| {
        *x = one / (one + (-*x).exp());
    });
    Ok(output)
}

/// Tanh (Hyperbolic Tangent) activation function
///
/// Applies the element-wise function: `f(x) = tanh(x)`
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// Array with tanh applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::tanh;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
/// let output = tanh(&input).expect("Tanh failed");
/// ```
pub fn tanh<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let mut output = input.clone();
    Zip::from(&mut output).for_each(|x| {
        *x = x.tanh();
    });
    Ok(output)
}

/// GELU (Gaussian Error Linear Unit) activation function
///
/// Applies the approximation: `f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// Array with GELU applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::gelu;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
/// let output = gelu(&input).expect("GELU failed");
/// ```
pub fn gelu<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let mut output = input.clone();
    let half = F::from(0.5).ok_or_else(|| {
        crate::error::NeuralError::ComputationError("Failed to convert constant".to_string())
    })?;
    let sqrt_2_over_pi = F::from(0.7978845608028654).ok_or_else(|| {
        crate::error::NeuralError::ComputationError("Failed to convert constant".to_string())
    })?;
    let coeff = F::from(0.044715).ok_or_else(|| {
        crate::error::NeuralError::ComputationError("Failed to convert constant".to_string())
    })?;
    let one = F::one();

    Zip::from(&mut output).for_each(|x| {
        let x3 = *x * *x * *x;
        let inner = sqrt_2_over_pi * (*x + coeff * x3);
        *x = half * *x * (one + inner.tanh());
    });
    Ok(output)
}

/// Leaky ReLU activation function
///
/// Applies the element-wise function: `f(x) = x if x > 0, else negative_slope * x`
///
/// # Arguments
///
/// * `input` - Input array
/// * `negative_slope` - Slope for negative values (typically 0.01)
///
/// # Returns
///
/// Array with Leaky ReLU applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::leaky_relu;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
/// let output = leaky_relu(&input, 0.01).expect("Leaky ReLU failed");
/// ```
pub fn leaky_relu<F: Float + Debug + NumAssign>(
    input: &Array<F, IxDyn>,
    negative_slope: F,
) -> Result<Array<F, IxDyn>> {
    let mut output = input.clone();
    let zero = F::zero();
    Zip::from(&mut output).for_each(|x| {
        if *x < zero {
            *x = negative_slope * *x;
        }
    });
    Ok(output)
}

/// Swish activation function (also known as SiLU)
///
/// Applies the element-wise function: `f(x) = x * sigmoid(x)`
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// Array with Swish applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::swish;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
/// let output = swish(&input).expect("Swish failed");
/// ```
pub fn swish<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let sigmoid_output = sigmoid(input)?;
    let mut output = input.clone();
    Zip::from(&mut output)
        .and(&sigmoid_output)
        .for_each(|x, &sig| {
            *x = *x * sig;
        });
    Ok(output)
}

/// Mish activation function
///
/// Applies the element-wise function: `f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// Array with Mish applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::mish;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
/// let output = mish(&input).expect("Mish failed");
/// ```
pub fn mish<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let mut output = input.clone();
    let one = F::one();
    Zip::from(&mut output).for_each(|x| {
        // softplus(x) = ln(1 + exp(x))
        let softplus = (one + x.exp()).ln();
        *x = *x * softplus.tanh();
    });
    Ok(output)
}

/// Softmax activation function
///
/// Applies softmax along the specified axis: `f(x_i) = exp(x_i) / sum(exp(x_j))`
///
/// # Arguments
///
/// * `input` - Input array
/// * `axis` - Axis along which to apply softmax (-1 for last axis)
///
/// # Returns
///
/// Array with softmax applied along the specified axis
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::softmax;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
/// let output = softmax(&input, -1).expect("Softmax failed");
/// ```
pub fn softmax<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>, axis: isize) -> Result<Array<F, IxDyn>> {
    // For simple 1D case or applying to last axis
    let mut output = input.clone();

    // Determine the actual axis
    let actual_axis = if axis < 0 {
        (input.ndim() as isize + axis) as usize
    } else {
        axis as usize
    };

    if actual_axis >= input.ndim() {
        return Err(crate::error::NeuralError::InvalidArgument(format!(
            "Axis {} out of bounds for array with {} dimensions",
            axis,
            input.ndim()
        )));
    }

    // Simple implementation for last axis
    if actual_axis == input.ndim() - 1 {
        // Find max for numerical stability
        let max_val = input.fold(F::neg_infinity(), |acc, &x| if x > acc { x } else { acc });

        // Subtract max and exponentiate
        Zip::from(&mut output).for_each(|x| {
            *x = (*x - max_val).exp();
        });

        // Normalize by sum
        let sum = output.sum();
        Zip::from(&mut output).for_each(|x| {
            *x = *x / sum;
        });
    } else {
        // For other axes, we need more sophisticated handling
        // This is a simplified version
        return Err(crate::error::NeuralError::InvalidArgument(
            "Softmax along non-last axis not yet implemented".to_string(),
        ));
    }

    Ok(output)
}

/// ELU (Exponential Linear Unit) activation function
///
/// Applies the element-wise function: `f(x) = x if x > 0, else alpha * (exp(x) - 1)`
///
/// # Arguments
///
/// * `input` - Input array
/// * `alpha` - Scaling factor for negative values (typically 1.0)
///
/// # Returns
///
/// Array with ELU applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::elu;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
/// let output = elu(&input, 1.0).expect("ELU failed");
/// ```
pub fn elu<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>, alpha: F) -> Result<Array<F, IxDyn>> {
    let mut output = input.clone();
    let zero = F::zero();
    let one = F::one();
    Zip::from(&mut output).for_each(|x| {
        if *x > zero {
            // x stays the same
        } else {
            *x = alpha * (x.exp() - one);
        }
    });
    Ok(output)
}

/// SELU (Scaled Exponential Linear Unit) activation function
///
/// Applies the element-wise function with self-normalizing properties:
/// `f(x) = scale * (x if x > 0, else alpha * (exp(x) - 1))`
///
/// Default values: scale = 1.0507, alpha = 1.67326
///
/// # Arguments
///
/// * `input` - Input array
///
/// # Returns
///
/// Array with SELU applied element-wise
///
/// # Examples
///
/// ```
/// use scirs2_neural::ops::activations::selu;
/// use scirs2_core::ndarray::Array;
///
/// let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
/// let output = selu(&input).expect("SELU failed");
/// ```
pub fn selu<F: Float + Debug + NumAssign>(input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
    let scale = F::from(1.0507009873554804934193349852946).ok_or_else(|| {
        crate::error::NeuralError::ComputationError("Failed to convert constant".to_string())
    })?;
    let alpha = F::from(1.6732632423543772848170429916717).ok_or_else(|| {
        crate::error::NeuralError::ComputationError("Failed to convert constant".to_string())
    })?;

    let mut output = input.clone();
    let zero = F::zero();
    let one = F::one();
    Zip::from(&mut output).for_each(|x| {
        if *x > zero {
            *x = scale * *x;
        } else {
            *x = scale * alpha * (x.exp() - one);
        }
    });
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_relu() {
        let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
        let output = relu(&input).expect("ReLU failed");
        let expected = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0]).into_dyn();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_sigmoid() {
        let input = Array::from_vec(vec![0.0]).into_dyn();
        let output = sigmoid(&input).expect("Sigmoid failed");
        assert!((output[[0]] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tanh() {
        let input = Array::from_vec(vec![0.0]).into_dyn();
        let output = tanh(&input).expect("Tanh failed");
        assert_eq!(output[[0]], 0.0);
    }

    #[test]
    fn test_gelu() {
        let input = Array::from_vec(vec![0.0]).into_dyn();
        let output = gelu(&input).expect("GELU failed");
        assert_eq!(output[[0]], 0.0);
    }

    #[test]
    fn test_leaky_relu() {
        let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
        let output = leaky_relu(&input, 0.01).expect("Leaky ReLU failed");
        assert!((output[[0]] - (-0.02)).abs() < 1e-6);
        assert!((output[[1]] - (-0.01)).abs() < 1e-6);
        assert_eq!(output[[2]], 0.0);
        assert_eq!(output[[3]], 1.0);
        assert_eq!(output[[4]], 2.0);
    }

    #[test]
    fn test_swish() {
        let input = Array::from_vec(vec![0.0]).into_dyn();
        let output = swish(&input).expect("Swish failed");
        assert_eq!(output[[0]], 0.0);
    }

    #[test]
    fn test_mish() {
        let input = Array::from_vec(vec![0.0]).into_dyn();
        let output = mish(&input).expect("Mish failed");
        assert_eq!(output[[0]], 0.0);
    }

    #[test]
    fn test_softmax_1d() {
        let input = Array::from_vec(vec![1.0, 2.0, 3.0]).into_dyn();
        let output = softmax(&input, -1).expect("Softmax failed");
        // Sum should be 1.0
        let sum: f64 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Values should be increasing
        assert!(output[[0]] < output[[1]]);
        assert!(output[[1]] < output[[2]]);
    }

    #[test]
    fn test_elu() {
        let input = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]).into_dyn();
        let output = elu(&input, 1.0).expect("ELU failed");
        // Positive values should remain unchanged
        assert_eq!(output[[3]], 1.0);
        assert_eq!(output[[4]], 2.0);
        // Negative values should be transformed
        assert!(output[[0]] < 0.0 && output[[0]] > -2.0);
    }

    #[test]
    fn test_selu() {
        let input = Array::from_vec(vec![-1.0, 0.0, 1.0]).into_dyn();
        let output = selu(&input).expect("SELU failed");
        // At 0, should be 0
        assert_eq!(output[[1]], 0.0);
        // Positive values should be scaled
        assert!(output[[2]] > 1.0);
    }
}
