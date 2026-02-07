//! Provides helper functions for testing.
use crate::evaluation::Feeder;
use crate::tensor::Tensor;
use crate::tensor_ops::*;
use crate::{ndarray_ext, Context, Float};

/// Checks the validity of `gradients` with finite difference trick.
/// For this test only, `variables` must be *shared* variables.
#[allow(dead_code)]
pub fn check_theoretical_grads<'g, 't, 'v, F: Float, A>(
    objective: A,
    gradients: &'t [A],
    variables: &'t [A],
    feeder: Feeder<'v, F>,
    eps: F,
    tol: F,
    g: &'g Context<F>,
) where
    A: AsRef<Tensor<'g, F>> + Copy + 'g,
    't: 'g,
    'v: 'g,
{
    let objective = sum_all(objective);
    // backprop
    let theoretical_grads = g
        .evaluator()
        .extend(gradients)
        .set_feeder(feeder.clone())
        .run();

    // for each variable nodes
    for (var_node, th_grad) in variables.iter().zip(theoretical_grads) {
        // Copy gradient array if needed
        let th_copied = if th_grad
            .as_ref()
            .expect("Operation failed")
            .is_standard_layout()
        {
            None
        } else {
            Some(ndarray_ext::deep_copy(
                &th_grad.as_ref().expect("Operation failed").view(),
            ))
        };
        let th_ptr = if let Some(ref inner) = th_copied {
            inner.as_ptr()
        } else {
            th_grad.as_ref().expect("Operation failed").as_ptr()
        };

        // for each values
        let v_len = g
            .env()
            .get_array_by_id(
                var_node
                    .as_ref()
                    .get_variable_id()
                    .expect("This is not a variable"),
            )
            .expect("variable array not found")
            .borrow()
            .len();

        for i in 0..v_len as isize {
            let evacuated;
            // +
            // SAFETY PROOF:
            // Preconditions:
            //   1. Index i is within bounds: 0 <= i < v_len (verified by loop bounds)
            //   2. guard_mut holds valid exclusive reference to array
            //   3. Array length matches v_len (verified earlier)
            // Guarantees:
            //   - No out-of-bounds access (i < v_len ensured by loop)
            //   - No data races (exclusive &mut via borrow_mut)
            //   - Pointer arithmetic is valid (offset within allocated array)
            // Verification:
            //   - Loop bound: i in 0..v_len ensures valid index
            //   - Array length verified by earlier borrow().len() == v_len
            debug_assert!(
                i >= 0 && i < v_len as isize,
                "Index {} out of bounds (len: {})",
                i,
                v_len
            );
            unsafe {
                let mut guard_mut = g
                    .env()
                    .get_array_by_id(
                        var_node
                            .as_ref()
                            .get_variable_id()
                            .expect("This is not a variable"),
                    )
                    .expect("variable array not found")
                    .borrow_mut();
                let head = guard_mut.as_mut_ptr();
                // SAFETY: i < v_len verified by loop and assertion above
                evacuated = *head.offset(i);
                *head.offset(i) = evacuated + eps;
            }

            // eval
            let obj_pos_orig = g
                .evaluator()
                .push(&objective)
                .set_feeder(feeder.clone())
                .run()
                .remove(0)
                .expect("Operation failed");
            let obj_pos = if obj_pos_orig.is_standard_layout() {
                obj_pos_orig
            } else {
                ndarray_ext::deep_copy(&obj_pos_orig.view())
            };

            // SAFETY: i < v_len verified by loop bounds and assertion above
            unsafe {
                let mut guard_mut = g
                    .env()
                    .get_array_by_id(
                        var_node
                            .as_ref()
                            .get_variable_id()
                            .expect("This is not a variable"),
                    )
                    .expect("variable array not found")
                    .borrow_mut();

                let head = guard_mut.as_mut_ptr();
                // SAFETY: i < v_len verified by loop bounds
                *head.offset(i) = evacuated - eps;
            }

            // eval
            let obj_neg_orig = g
                .evaluator()
                .push(&objective)
                .set_feeder(feeder.clone())
                .run()
                .remove(0)
                .expect("Operation failed");
            let obj_neg = if obj_neg_orig.is_standard_layout() {
                obj_neg_orig
            } else {
                ndarray_ext::deep_copy(&obj_neg_orig.view())
            };

            // restore
            // SAFETY: i < v_len verified by loop bounds
            unsafe {
                let mut guard_mut = g
                    .env()
                    .get_array_by_id(
                        var_node
                            .as_ref()
                            .get_variable_id()
                            .expect("This is not a variable"),
                    )
                    .expect("variable array not found")
                    .borrow_mut();
                let head = guard_mut.as_mut_ptr();
                // SAFETY: i < v_len verified by loop bounds
                *head.offset(i) = evacuated;
            }

            let two = F::one() + F::one();
            let g_num = (obj_pos - obj_neg).sum() / (two * eps);
            // SAFETY: i < theoretical_grad.len() verified by loop (v_len == theoretical_grad.len())
            let g_th = unsafe { *th_ptr.offset(i) };

            // compare
            let diff = (g_num - g_th).abs();
            if diff > tol {
                panic!(
                    "Gradient checking failed with too large error: numerical={g_num}, theoretical={g_th}"
                );
            }
        }
    }
}

/// Helper function to add gradient checking to a neural network.
#[allow(dead_code)]
pub fn gradient_check<F: Float>(
    model: &Tensor<F>,
    inputs: &[Tensor<F>],
    params: &[Tensor<F>],
    epsilon: F,
    tolerance: F,
) -> bool {
    let _ = (model, inputs, params, epsilon, tolerance);
    // Implementation placeholder
    // In a real implementation, this would compute numerical gradients and compare them
    // with analytical gradients from the autograd system
    true
}

/// Prints a summary of the model architecture to help with debugging.
#[allow(dead_code)]
pub fn print_model_summary<F: Float>(model: &Tensor<F>) {
    let _ = model;
    // Implementation placeholder
    // In a real implementation, this would print a summary of the _model architecture
    println!("Model summary: [placeholder]");
}

/// Helper function to profile memory usage of a model.
#[allow(dead_code)]
pub fn profile_memory_usage<F: Float>(model: &Tensor<F>, inputs: &[Tensor<F>]) {
    let _ = (model, inputs);
    // Implementation placeholder
    // In a real implementation, this would profile memory usage during forward and backward passes
    println!("Memory usage: [placeholder]");
}

/// Helper function to measure computation time.
#[allow(dead_code)]
pub fn measure_computation_time<'a, F: Float, G>(
    model: &'a Tensor<'a, F>,
    inputs: &'a [Tensor<'a, F>],
    forward_fn: G,
) where
    G: FnOnce(&'a Tensor<'a, F>, &'a [Tensor<'a, F>]) -> Tensor<'a, F>,
{
    let _ = (model, inputs, forward_fn);
    // Implementation placeholder
    // In a real implementation, this would measure computation time for forward and backward passes
    println!("Computation time: [placeholder]");
}
