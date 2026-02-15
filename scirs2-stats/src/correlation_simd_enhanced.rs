//! Enhanced SIMD-optimized correlation and covariance functions for v0.2.0
//!
//! This module provides comprehensive SIMD-accelerated implementations of
//! correlation and covariance computations with improved performance.

use crate::descriptive_simd::mean_simd;
use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::simd_ops::{AutoOptimizer, SimdUnifiedOps};

/// Compute covariance matrix using SIMD operations
///
/// This function efficiently computes the covariance matrix for multiple variables
/// using SIMD acceleration for improved performance (2-3x speedup expected).
///
/// # Arguments
///
/// * `data` - 2D array where each row is an observation and each column is a variable
/// * `rowvar` - If true, rows are variables and columns are observations
/// * `ddof` - Delta degrees of freedom (default: 1 for sample covariance)
///
/// # Returns
///
/// Covariance matrix
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::correlation_simd_enhanced::covariance_matrix_simd;
///
/// let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let cov = covariance_matrix_simd(&data.view(), false, 1).expect("Computation failed");
/// assert_eq!(cov.shape(), &[3, 3]);
/// ```
pub fn covariance_matrix_simd<F, D>(
    data: &ArrayBase<D, Ix2>,
    rowvar: bool,
    ddof: usize,
) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    let (n_vars, n_obs) = if rowvar {
        (data.nrows(), data.ncols())
    } else {
        (data.ncols(), data.nrows())
    };

    if n_obs <= ddof {
        return Err(StatsError::invalid_argument(
            "Not enough observations for the given degrees of freedom",
        ));
    }

    let optimizer = AutoOptimizer::new();
    let mut cov_matrix = Array2::zeros((n_vars, n_vars));

    // Compute means for each variable
    let means: Vec<F> = (0..n_vars)
        .map(|i| {
            let var = if rowvar {
                data.slice(s![i, ..])
            } else {
                data.slice(s![.., i])
            };
            mean_simd(&var).unwrap_or_else(|_| F::zero())
        })
        .collect();

    // Compute covariance matrix
    for i in 0..n_vars {
        for j in i..n_vars {
            let var_i = if rowvar {
                data.slice(s![i, ..])
            } else {
                data.slice(s![.., i])
            };

            let var_j = if rowvar {
                data.slice(s![j, ..])
            } else {
                data.slice(s![.., j])
            };

            let cov = if optimizer.should_use_simd(n_obs) {
                // SIMD path
                let mean_i_array = Array1::from_elem(n_obs, means[i]);
                let mean_j_array = Array1::from_elem(n_obs, means[j]);

                let dev_i = F::simd_sub(&var_i, &mean_i_array.view());
                let dev_j = F::simd_sub(&var_j, &mean_j_array.view());
                let products = F::simd_mul(&dev_i.view(), &dev_j.view());
                let sum_products = F::simd_sum(&products.view());

                sum_products / F::from(n_obs - ddof).unwrap_or_else(|| F::one())
            } else {
                // Scalar fallback
                let mut sum = F::zero();
                for k in 0..n_obs {
                    let dev_i = var_i[k] - means[i];
                    let dev_j = var_j[k] - means[j];
                    sum = sum + dev_i * dev_j;
                }
                sum / F::from(n_obs - ddof).unwrap_or_else(|| F::one())
            };

            cov_matrix[(i, j)] = cov;
            if i != j {
                cov_matrix[(j, i)] = cov; // Symmetric matrix
            }
        }
    }

    Ok(cov_matrix)
}

/// Compute Spearman rank correlation using SIMD operations
///
/// This function computes the Spearman rank correlation coefficient using
/// SIMD acceleration after converting values to ranks.
///
/// # Arguments
///
/// * `x` - First input array
/// * `y` - Second input array
///
/// # Returns
///
/// Spearman's rank correlation coefficient
pub fn spearman_r_simd<F, D>(x: &ArrayBase<D, Ix1>, y: &ArrayBase<D, Ix1>) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    if x.len() != y.len() {
        return Err(StatsError::dimension_mismatch(
            "Arrays must have the same length",
        ));
    }

    if x.is_empty() {
        return Err(StatsError::invalid_argument("Arrays cannot be empty"));
    }

    let n = x.len();

    // Convert to ranks
    let rank_x = compute_ranks(x);
    let rank_y = compute_ranks(y);

    // Use Pearson correlation on ranks
    let mean_rx = mean_simd(&rank_x.view())?;
    let mean_ry = mean_simd(&rank_y.view())?;

    let optimizer = AutoOptimizer::new();

    if optimizer.should_use_simd(n) {
        // SIMD path
        let mean_rx_array = Array1::from_elem(n, mean_rx);
        let mean_ry_array = Array1::from_elem(n, mean_ry);

        let rx_dev = F::simd_sub(&rank_x.view(), &mean_rx_array.view());
        let ry_dev = F::simd_sub(&rank_y.view(), &mean_ry_array.view());

        let xy_dev = F::simd_mul(&rx_dev.view(), &ry_dev.view());
        let rx_dev_sq = F::simd_mul(&rx_dev.view(), &rx_dev.view());
        let ry_dev_sq = F::simd_mul(&ry_dev.view(), &ry_dev.view());

        let sum_xy = F::simd_sum(&xy_dev.view());
        let sum_rx2 = F::simd_sum(&rx_dev_sq.view());
        let sum_ry2 = F::simd_sum(&ry_dev_sq.view());

        if sum_rx2 <= F::epsilon() || sum_ry2 <= F::epsilon() {
            return Err(StatsError::invalid_argument(
                "Cannot compute correlation when one or both variables have zero variance",
            ));
        }

        let corr = sum_xy / (sum_rx2 * sum_ry2).sqrt();
        Ok(corr.max(-F::one()).min(F::one()))
    } else {
        // Scalar fallback
        let mut sum_xy = F::zero();
        let mut sum_rx2 = F::zero();
        let mut sum_ry2 = F::zero();

        for i in 0..n {
            let rx_dev = rank_x[i] - mean_rx;
            let ry_dev = rank_y[i] - mean_ry;

            sum_xy = sum_xy + rx_dev * ry_dev;
            sum_rx2 = sum_rx2 + rx_dev * rx_dev;
            sum_ry2 = sum_ry2 + ry_dev * ry_dev;
        }

        if sum_rx2 <= F::epsilon() || sum_ry2 <= F::epsilon() {
            return Err(StatsError::invalid_argument(
                "Cannot compute correlation when one or both variables have zero variance",
            ));
        }

        let corr = sum_xy / (sum_rx2 * sum_ry2).sqrt();
        Ok(corr.max(-F::one()).min(F::one()))
    }
}

/// Helper function to compute ranks
fn compute_ranks<F, D>(data: &ArrayBase<D, Ix1>) -> Array1<F>
where
    F: Float + NumCast,
    D: Data<Elem = F>,
{
    let n = data.len();
    let mut indexed: Vec<(usize, F)> = data.iter().copied().enumerate().collect();

    // Sort by value
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks (average rank for ties)
    let mut ranks = Array1::zeros(n);
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Find the end of tied values
        while j < n && (indexed[j].1 - indexed[i].1).abs() < F::epsilon() {
            j += 1;
        }

        // Compute average rank for tied values
        let avg_rank = F::from((i + j + 1) as f64 / 2.0).unwrap_or_else(|| F::zero());

        // Assign average rank to all tied values
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }

        i = j;
    }

    ranks
}

/// Compute partial correlation using SIMD operations
///
/// Computes the partial correlation between x and y, controlling for z.
///
/// # Arguments
///
/// * `x` - First variable
/// * `y` - Second variable
/// * `z` - Controlling variables (can be multivariate)
///
/// # Returns
///
/// Partial correlation coefficient
pub fn partial_correlation_simd<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    z: &ArrayView2<F>,
) -> StatsResult<F>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if x.len() != y.len() || x.len() != z.nrows() {
        return Err(StatsError::dimension_mismatch(
            "All arrays must have compatible dimensions",
        ));
    }

    // Use the formula: partial_corr(x,y|z) = (corr(x,y) - corr(x,z)*corr(y,z)) / sqrt((1-corr(x,z)^2)*(1-corr(y,z)^2))
    // For simplicity, we'll compute residuals after regressing x and y on z

    // This is a simplified implementation - full implementation would use
    // regression to compute residuals
    use crate::correlation_simd::pearson_r_simd;

    // For single controlling variable
    if z.ncols() == 1 {
        let z_col = z.column(0);
        let rxy = pearson_r_simd(x, y)?;
        let rxz = pearson_r_simd(x, &z_col)?;
        let ryz = pearson_r_simd(y, &z_col)?;

        let numerator = rxy - rxz * ryz;
        let denominator = ((F::one() - rxz * rxz) * (F::one() - ryz * ryz)).sqrt();

        if denominator <= F::epsilon() {
            return Err(StatsError::invalid_argument(
                "Cannot compute partial correlation - controlling variable perfectly predicts x or y",
            ));
        }

        Ok(numerator / denominator)
    } else {
        // For multiple controlling variables, use regression residuals
        // This is a placeholder - full implementation would require matrix operations
        Err(StatsError::not_implemented(
            "Partial correlation with multiple controlling variables not yet implemented",
        ))
    }
}

/// Compute rolling correlation using SIMD operations
///
/// Computes correlation between x and y over a rolling window.
///
/// # Arguments
///
/// * `x` - First input array
/// * `y` - Second input array
/// * `window_size` - Size of the rolling window
///
/// # Returns
///
/// Array of rolling correlation values
pub fn rolling_correlation_simd<F, D>(
    x: &ArrayBase<D, Ix1>,
    y: &ArrayBase<D, Ix1>,
    window_size: usize,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    if x.len() != y.len() {
        return Err(StatsError::dimension_mismatch(
            "Arrays must have the same length",
        ));
    }

    if window_size < 2 {
        return Err(StatsError::invalid_argument(
            "Window size must be at least 2",
        ));
    }

    let n = x.len();
    if n < window_size {
        return Err(StatsError::invalid_argument(
            "Array length must be at least window size",
        ));
    }

    let n_windows = n - window_size + 1;
    let mut result = Array1::zeros(n_windows);

    use crate::correlation_simd::pearson_r_simd;

    for i in 0..n_windows {
        let x_window = x.slice(s![i..i + window_size]);
        let y_window = y.slice(s![i..i + window_size]);
        result[i] = pearson_r_simd(&x_window, &y_window)?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_covariance_matrix_simd() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let cov = covariance_matrix_simd(&data.view(), false, 1).expect("Failed");

        // Check symmetry
        assert_abs_diff_eq!(cov[(0, 1)], cov[(1, 0)], epsilon = 1e-10);

        // Check diagonal (variances are positive)
        assert!(cov[(0, 0)] > 0.0);
        assert!(cov[(1, 1)] > 0.0);
    }

    #[test]
    fn test_spearman_r_simd() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 4.0, 3.0, 2.0, 1.0];

        let rho = spearman_r_simd(&x.view(), &y.view()).expect("Failed");
        assert_abs_diff_eq!(rho, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rolling_correlation_simd() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = array![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let rolling_corr = rolling_correlation_simd(&x.view(), &y.view(), 3).expect("Failed");
        assert_eq!(rolling_corr.len(), 8);
    }
}
