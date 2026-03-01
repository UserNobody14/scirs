//! One-dimensional interpolation methods
//!
//! This module provides functionality for interpolating one-dimensional data.

mod basic_interp;
pub mod monotonic;
pub mod pchip;

// Re-export interpolation functions
pub use basic_interp::{cubic_interpolate, linear_interpolate, nearest_interpolate};
pub use monotonic::{
    hyman_interpolate, modified_akima_interpolate, monotonic_interpolate, steffen_interpolate,
    MonotonicInterpolator, MonotonicMethod,
};
pub use pchip::{pchip_interpolate, PchipInterpolator};

use crate::error::{InterpolateError, InterpolateResult};
use crate::spline::CubicSpline;
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

/// Available interpolation methods
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum InterpolationMethod {
    /// Nearest neighbor interpolation
    Nearest,
    /// Linear interpolation
    #[default]
    Linear,
    /// Cubic interpolation
    Cubic,
    /// PCHIP interpolation (monotonic)
    Pchip,
}

/// Options for extrapolation behavior
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ExtrapolateMode {
    /// Return error when extrapolating
    #[default]
    Error,
    /// Extrapolate using the interpolation method
    Extrapolate,
    /// Use nearest valid value
    Nearest,
}

/// One-dimensional interpolation object
///
/// Provides a way to interpolate values at arbitrary points within a range
/// based on a set of known x and y values.
#[derive(Debug, Clone)]
pub struct Interp1d<F: Float> {
    /// X coordinates (must be sorted)
    x: Array1<F>,
    /// Y coordinates
    y: Array1<F>,
    /// Interpolation method
    method: InterpolationMethod,
    /// Extrapolation mode
    extrapolate: ExtrapolateMode,
}

impl<F: InterpolationFloat + ToString> Interp1d<F> {
    /// Create a new interpolation object
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    /// * `method` - The interpolation method to use
    /// * `extrapolate` - The extrapolation behavior
    ///
    /// # Returns
    ///
    /// A new `Interp1d` object
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_core::ndarray::array;
    /// use scirs2_interpolate::interp1d::{Interp1d, InterpolationMethod, ExtrapolateMode};
    ///
    /// let x = array![0.0f64, 1.0, 2.0, 3.0];
    /// let y = array![0.0f64, 1.0, 4.0, 9.0];
    ///
    /// // Create a linear interpolator
    /// let interp = Interp1d::new(
    ///     &x.view(), &y.view(),
    ///     InterpolationMethod::Linear,
    ///     ExtrapolateMode::Error
    /// ).expect("Operation failed");
    ///
    /// // Interpolate at x = 1.5
    /// let y_interp = interp.evaluate(1.5);
    /// assert!(y_interp.is_ok());
    /// assert!((y_interp.expect("Operation failed") - 2.5).abs() < 1e-10);
    /// ```
    pub fn new(
        x: &ArrayView1<F>,
        y: &ArrayView1<F>,
        method: InterpolationMethod,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 2 {
            return Err(InterpolateError::insufficient_points(
                2,
                x.len(),
                "interpolation",
            ));
        }

        // Check for NaN and Infinity in input data
        for i in 0..x.len() {
            if !x[i].is_finite() {
                return Err(InterpolateError::invalid_input(format!(
                    "x values must be finite, found non-finite value at index {}",
                    i
                )));
            }
            if !y[i].is_finite() {
                return Err(InterpolateError::invalid_input(format!(
                    "y values must be finite, found non-finite value at index {}",
                    i
                )));
            }
        }

        // Check that x is sorted
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x values must be sorted in ascending order".to_string(),
                ));
            }
        }

        // For cubic interpolation, need at least 2 points (quadratic fallback for 3, linear for 2)
        if method == InterpolationMethod::Cubic && x.len() < 2 {
            return Err(InterpolateError::insufficient_points(
                2,
                x.len(),
                "cubic interpolation",
            ));
        }

        Ok(Interp1d {
            x: x.to_owned(),
            y: y.to_owned(),
            method,
            extrapolate,
        })
    }

    /// Evaluate the interpolation at the given points
    ///
    /// # Arguments
    ///
    /// * `xnew` - The x coordinate at which to evaluate the interpolation
    ///
    /// # Returns
    ///
    /// The interpolated y value at `xnew`
    pub fn evaluate(&self, xnew: F) -> InterpolateResult<F> {
        // Check if we're extrapolating
        let is_extrapolating = xnew < self.x[0] || xnew > self.x[self.x.len() - 1];

        if is_extrapolating {
            match self.extrapolate {
                ExtrapolateMode::Error => {
                    return Err(InterpolateError::out_of_domain_with_suggestion(
                        xnew,
                        self.x[0],
                        self.x[self.x.len() - 1],
                        "1D interpolation evaluation",
                        format!("Use ExtrapolateMode::Extrapolate for linear extrapolation, ExtrapolateMode::Nearest for constant extrapolation, or ensure query points are within the data range [{:?}, {:?}]", 
                               self.x[0], self.x[self.x.len() - 1])
                    ));
                }
                ExtrapolateMode::Nearest => {
                    if xnew < self.x[0] {
                        return Ok(self.y[0]);
                    } else {
                        return Ok(self.y[self.y.len() - 1]);
                    }
                }
                ExtrapolateMode::Extrapolate => {
                    // PCHIP and Cubic use polynomial continuation via their own evaluators
                    if self.method == InterpolationMethod::Pchip {
                        let pchip = PchipInterpolator::new(&self.x.view(), &self.y.view(), true)?;
                        return pchip.evaluate(xnew);
                    }
                    if self.method == InterpolationMethod::Cubic {
                        let spline =
                            CubicSpline::new_not_a_knot(&self.x.view(), &self.y.view())?;
                        let n = self.x.len();
                        let seg = if xnew < self.x[0] { 0 } else { n - 2 };
                        let dx = xnew - self.x[seg];
                        let c = spline.coeffs();
                        let result = c[[seg, 0]]
                            + c[[seg, 1]] * dx
                            + c[[seg, 2]] * dx * dx
                            + c[[seg, 3]] * dx * dx * dx;
                        return Ok(result);
                    }
                    // For other methods, use linear extrapolation based on edge segments
                    if xnew < self.x[0] {
                        let x0 = self.x[0];
                        let x1 = self.x[1];
                        let y0 = self.y[0];
                        let y1 = self.y[1];
                        let slope = (y1 - y0) / (x1 - x0);
                        return Ok(y0 + (xnew - x0) * slope);
                    } else {
                        let n = self.x.len();
                        let x0 = self.x[n - 2];
                        let x1 = self.x[n - 1];
                        let y0 = self.y[n - 2];
                        let y1 = self.y[n - 1];
                        let slope = (y1 - y0) / (x1 - x0);
                        return Ok(y1 + (xnew - x1) * slope);
                    }
                }
            }
        }

        // Find the index of the segment containing xnew using binary search
        let idx = self.find_segment(xnew);

        // Special case: xnew is exactly the last point
        if xnew == self.x[self.x.len() - 1] {
            return Ok(self.y[self.x.len() - 1]);
        }

        // Apply the selected interpolation method
        match self.method {
            InterpolationMethod::Nearest => {
                nearest_interp(&self.x.view(), &self.y.view(), idx, xnew)
            }
            InterpolationMethod::Linear => linear_interp(&self.x.view(), &self.y.view(), idx, xnew),
            InterpolationMethod::Cubic => {
                let spline =
                    CubicSpline::new_not_a_knot(&self.x.view(), &self.y.view())?;
                spline.evaluate(xnew)
            }
            InterpolationMethod::Pchip => {
                // For PCHIP, we'll create a PCHIP interpolator and use it
                // This is not the most efficient approach, but it keeps the interface consistent
                let extrapolate = self.extrapolate == ExtrapolateMode::Extrapolate
                    || self.extrapolate == ExtrapolateMode::Nearest;
                let pchip = PchipInterpolator::new(&self.x.view(), &self.y.view(), extrapolate)?;
                pchip.evaluate(xnew)
            }
        }
    }

    /// Find the segment index containing the given x value using binary search.
    ///
    /// Returns the index `i` such that `x[i] <= xnew <= x[i+1]`.
    /// For values at or beyond the last point, returns `x.len() - 2`.
    fn find_segment(&self, xnew: F) -> usize {
        let n = self.x.len();
        if n < 2 {
            return 0;
        }

        // Binary search: find the largest i such that x[i] <= xnew
        let mut lo = 0usize;
        let mut hi = n - 1;

        // Clamp to valid range
        if xnew <= self.x[0] {
            return 0;
        }
        if xnew >= self.x[n - 1] {
            return n - 2;
        }

        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if self.x[mid] <= xnew {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        lo
    }

    /// Evaluate the interpolation at multiple points
    ///
    /// # Arguments
    ///
    /// * `xnew` - The x coordinates at which to evaluate the interpolation
    ///
    /// # Returns
    ///
    /// The interpolated y values at `xnew`
    pub fn evaluate_array(&self, xnew: &ArrayView1<F>) -> InterpolateResult<Array1<F>> {
        // For spline-based methods, build once and evaluate many times
        if self.method == InterpolationMethod::Cubic {
            let spline = CubicSpline::new_not_a_knot(&self.x.view(), &self.y.view())?;
            let mut result = Array1::zeros(xnew.len());
            for (i, &xv) in xnew.iter().enumerate() {
                let is_extrap = xv < self.x[0] || xv > self.x[self.x.len() - 1];
                if is_extrap {
                    match self.extrapolate {
                        ExtrapolateMode::Error => {
                            return Err(InterpolateError::out_of_domain_with_suggestion(
                                xv,
                                self.x[0],
                                self.x[self.x.len() - 1],
                                "1D interpolation evaluation",
                                format!("Use ExtrapolateMode::Extrapolate or ensure query points are within [{:?}, {:?}]",
                                    self.x[0], self.x[self.x.len() - 1])
                            ));
                        }
                        ExtrapolateMode::Nearest => {
                            result[i] = if xv < self.x[0] {
                                self.y[0]
                            } else {
                                self.y[self.y.len() - 1]
                            };
                        }
                        ExtrapolateMode::Extrapolate => {
                            let n = self.x.len();
                            let seg = if xv < self.x[0] { 0 } else { n - 2 };
                            let dx = xv - self.x[seg];
                            let c = spline.coeffs();
                            result[i] = c[[seg, 0]]
                                + c[[seg, 1]] * dx
                                + c[[seg, 2]] * dx * dx
                                + c[[seg, 3]] * dx * dx * dx;
                        }
                    }
                } else if xv == self.x[self.x.len() - 1] {
                    result[i] = self.y[self.y.len() - 1];
                } else {
                    result[i] = spline.evaluate(xv)?;
                }
            }
            return Ok(result);
        }

        let mut result = Array1::zeros(xnew.len());
        for (i, &x) in xnew.iter().enumerate() {
            result[i] = self.evaluate(x)?;
        }
        Ok(result)
    }
}

/// Perform nearest neighbor interpolation
///
/// # Arguments
///
/// * `x` - The x coordinates
/// * `y` - The y coordinates
/// * `idx` - The index of the segment containing the target point
/// * `xnew` - The x coordinate at which to interpolate
///
/// # Returns
///
/// The interpolated value
#[allow(dead_code)]
fn nearest_interp<F: Float>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    idx: usize,
    xnew: F,
) -> InterpolateResult<F> {
    // Find which of the two points is closer
    let dist_left = (xnew - x[idx]).abs();
    let dist_right = (xnew - x[idx + 1]).abs();

    if dist_left <= dist_right {
        Ok(y[idx])
    } else {
        Ok(y[idx + 1])
    }
}

/// Perform linear interpolation
///
/// # Arguments
///
/// * `x` - The x coordinates
/// * `y` - The y coordinates
/// * `idx` - The index of the segment containing the target point
/// * `xnew` - The x coordinate at which to interpolate
///
/// # Returns
///
/// The interpolated value
#[allow(dead_code)]
fn linear_interp<F: Float>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    idx: usize,
    xnew: F,
) -> InterpolateResult<F> {
    let x0 = x[idx];
    let x1 = x[idx + 1];
    let y0 = y[idx];
    let y1 = y[idx + 1];

    // Avoid division by zero
    if x0 == x1 {
        return Ok(y0); // or y1, they should be the same
    }

    // Linear interpolation formula: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    Ok(y0 + (xnew - x0) * (y1 - y0) / (x1 - x0))
}

/// Perform cubic interpolation
///
/// # Arguments
///
/// * `x` - The x coordinates
/// * `y` - The y coordinates
/// * `idx` - The index of the segment containing the target point
/// * `xnew` - The x coordinate at which to interpolate
///
/// # Returns
///
/// The interpolated value
#[allow(dead_code)]
fn cubic_interp<F: Float + FromPrimitive>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    idx: usize,
    xnew: F,
) -> InterpolateResult<F> {
    // We need 4 points for cubic interpolation
    // If we're near the edges, we need to adjust the indices
    let (i0, i1, i2, i3) = if idx == 0 {
        (0, 0, 1, 2)
    } else if idx == x.len() - 2 {
        (idx - 1, idx, idx + 1, idx + 1)
    } else {
        // Handles both idx == x.len() - 3 and idx > x.len() - 3 cases since they're identical
        (idx - 1, idx, idx + 1, idx + 2)
    };

    let _x0 = x[i0];
    let x1 = x[i1];
    let x2 = x[i2];
    let _x3 = x[i3];

    let y0 = y[i0];
    let y1 = y[i1];
    let y2 = y[i2];
    let y3 = y[i3];

    // Normalized position within the interval [x1, x2]
    let t = if x2 != x1 {
        (xnew - x1) / (x2 - x1)
    } else {
        F::zero()
    };

    // Calculate cubic interpolation using Catmull-Rom spline
    // p(t) = 0.5 * ((2*p1) +
    //               (-p0 + p2) * t +
    //               (2*p0 - 5*p1 + 4*p2 - p3) * t^2 +
    //               (-p0 + 3*p1 - 3*p2 + p3) * t^3)

    let two = F::from_f64(2.0).expect("Operation failed");
    let three = F::from_f64(3.0).expect("Operation failed");
    let four = F::from_f64(4.0).expect("Operation failed");
    let five = F::from_f64(5.0).expect("Operation failed");
    let half = F::from_f64(0.5).expect("Operation failed");

    let t2 = t * t;
    let t3 = t2 * t;

    let c0 = two * y1;
    let c1 = -y0 + y2;
    let c2 = two * y0 - five * y1 + four * y2 - y3;
    let c3 = -y0 + three * y1 - three * y2 + y3;

    let result = half * (c0 + c1 * t + c2 * t2 + c3 * t3);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_nearest_interpolation() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let interp = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Nearest,
            ExtrapolateMode::Error,
        )
        .expect("Operation failed");

        // Test points exactly at data points
        assert_relative_eq!(interp.evaluate(0.0).expect("Operation failed"), 0.0);
        assert_relative_eq!(interp.evaluate(1.0).expect("Operation failed"), 1.0);
        assert_relative_eq!(interp.evaluate(2.0).expect("Operation failed"), 4.0);
        assert_relative_eq!(interp.evaluate(3.0).expect("Operation failed"), 9.0);

        // Test points between data points
        assert_relative_eq!(interp.evaluate(0.4).expect("Operation failed"), 0.0);
        assert_relative_eq!(interp.evaluate(0.6).expect("Operation failed"), 1.0);
        assert_relative_eq!(interp.evaluate(1.4).expect("Operation failed"), 1.0);
        assert_relative_eq!(interp.evaluate(1.6).expect("Operation failed"), 4.0);
    }

    #[test]
    fn test_linear_interpolation() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let interp = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Linear,
            ExtrapolateMode::Error,
        )
        .expect("Operation failed");

        // Test points exactly at data points
        assert_relative_eq!(interp.evaluate(0.0).expect("Operation failed"), 0.0);
        assert_relative_eq!(interp.evaluate(1.0).expect("Operation failed"), 1.0);
        assert_relative_eq!(interp.evaluate(2.0).expect("Operation failed"), 4.0);
        assert_relative_eq!(interp.evaluate(3.0).expect("Operation failed"), 9.0);

        // Test points between data points
        assert_relative_eq!(interp.evaluate(0.5).expect("Operation failed"), 0.5);
        assert_relative_eq!(interp.evaluate(1.5).expect("Operation failed"), 2.5);
        assert_relative_eq!(interp.evaluate(2.5).expect("Operation failed"), 6.5);
    }

    #[test]
    fn test_cubic_interpolation() {
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![0.0, 1.0, 4.0, 9.0, 16.0];

        let interp = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Cubic,
            ExtrapolateMode::Error,
        )
        .expect("Operation failed");

        // Test points exactly at data points
        assert_relative_eq!(interp.evaluate(0.0).unwrap(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(interp.evaluate(1.0).unwrap(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(interp.evaluate(2.0).unwrap(), 4.0, epsilon = 1e-12);
        assert_relative_eq!(interp.evaluate(3.0).unwrap(), 9.0, epsilon = 1e-12);
        assert_relative_eq!(interp.evaluate(4.0).unwrap(), 16.0, epsilon = 1e-12);

        // Not-a-knot cubic spline reproduces quadratic y=x^2 exactly
        // (scipy reference: interp1d(x, y, kind='cubic'))
        assert_relative_eq!(interp.evaluate(0.5).unwrap(), 0.25, epsilon = 1e-10);
        assert_relative_eq!(interp.evaluate(1.5).unwrap(), 2.25, epsilon = 1e-10);
        assert_relative_eq!(interp.evaluate(2.5).unwrap(), 6.25, epsilon = 1e-10);
        assert_relative_eq!(interp.evaluate(3.5).unwrap(), 12.25, epsilon = 1e-10);
    }

    #[test]
    fn test_cubic_reproduces_linear() {
        // Not-a-knot cubic spline must reproduce linear data exactly
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 3.0, 5.0, 7.0, 9.0]; // y = 2x + 1

        let interp = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Cubic,
            ExtrapolateMode::Error,
        )
        .unwrap();

        assert_relative_eq!(interp.evaluate(0.5).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(interp.evaluate(1.5).unwrap(), 4.0, epsilon = 1e-10);
        assert_relative_eq!(interp.evaluate(2.5).unwrap(), 6.0, epsilon = 1e-10);
        assert_relative_eq!(interp.evaluate(3.5).unwrap(), 8.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cubic_nonuniform_grid() {
        // Validated against scipy CubicSpline(x, y, bc_type='not-a-knot')
        let x = array![0.0, 0.3, 0.8, 1.5, 2.0, 3.0, 3.7, 4.0];
        let y = array![
            0.0_f64.sin(),
            0.3_f64.sin(),
            0.8_f64.sin(),
            1.5_f64.sin(),
            2.0_f64.sin(),
            3.0_f64.sin(),
            3.7_f64.sin(),
            4.0_f64.sin()
        ];

        let interp = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Cubic,
            ExtrapolateMode::Error,
        )
        .unwrap();

        // scipy reference values
        assert_relative_eq!(interp.evaluate(0.1).unwrap(), 0.099910627848514, epsilon = 1e-10);
        assert_relative_eq!(interp.evaluate(0.5).unwrap(), 0.479425652347223, epsilon = 1e-10);
        assert_relative_eq!(interp.evaluate(1.0).unwrap(), 0.840731841607741, epsilon = 1e-10);
        assert_relative_eq!(interp.evaluate(2.5).unwrap(), 0.595388243453342, epsilon = 1e-10);
        assert_relative_eq!(interp.evaluate(3.5).unwrap(), -0.349966440965012, epsilon = 1e-10);
    }

    #[test]
    fn test_cubic_3_points() {
        // n=3: falls back to quadratic fit, matching scipy
        let x = array![0.0, 1.0, 3.0];
        let y = array![1.0, 2.0, 0.0];

        let interp = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Cubic,
            ExtrapolateMode::Error,
        )
        .unwrap();

        // scipy CubicSpline([0,1,3],[1,2,0], bc_type='not-a-knot')
        assert_relative_eq!(interp.evaluate(0.5).unwrap(), 1.666666666666667, epsilon = 1e-10);
        assert_relative_eq!(interp.evaluate(1.5).unwrap(), 2.0, epsilon = 1e-10);
        assert_relative_eq!(interp.evaluate(2.0).unwrap(), 1.666666666666667, epsilon = 1e-10);
    }

    #[test]
    fn test_pchip_interpolation() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        let interp = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Pchip,
            ExtrapolateMode::Error,
        )
        .expect("Operation failed");

        // Test points exactly at data points
        assert_relative_eq!(interp.evaluate(0.0).expect("Operation failed"), 0.0);
        assert_relative_eq!(interp.evaluate(1.0).expect("Operation failed"), 1.0);
        assert_relative_eq!(interp.evaluate(2.0).expect("Operation failed"), 4.0);
        assert_relative_eq!(interp.evaluate(3.0).expect("Operation failed"), 9.0);

        // For this monotonically increasing dataset,
        // PCHIP should preserve monotonicity
        let y_05 = interp.evaluate(0.5).expect("Operation failed");
        let y_15 = interp.evaluate(1.5).expect("Operation failed");
        let y_25 = interp.evaluate(2.5).expect("Operation failed");

        assert!(y_05 > 0.0 && y_05 < 1.0);
        assert!(y_15 > 1.0 && y_15 < 4.0);
        assert!(y_25 > 4.0 && y_25 < 9.0);
    }

    #[test]
    fn test_extrapolation_modes() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];

        // Test error mode
        let interp_error = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Linear,
            ExtrapolateMode::Error,
        )
        .expect("Operation failed");

        assert!(interp_error.evaluate(-1.0).is_err());
        assert!(interp_error.evaluate(4.0).is_err());

        // Test nearest mode
        let interp_nearest = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Linear,
            ExtrapolateMode::Nearest,
        )
        .expect("Operation failed");

        assert_relative_eq!(
            interp_nearest.evaluate(-1.0).expect("Operation failed"),
            0.0
        );
        assert_relative_eq!(interp_nearest.evaluate(4.0).expect("Operation failed"), 9.0);

        // Test extrapolate mode
        let interp_extrapolate = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Linear,
            ExtrapolateMode::Extrapolate,
        )
        .expect("Operation failed");

        // For this data, the linear extrapolation is based on the slope of the segments
        // For x=-1.0, we use the first segment (0,0) - (1,1) which has slope 1
        assert_relative_eq!(
            interp_extrapolate.evaluate(-1.0).expect("Operation failed"),
            -1.0
        );

        // For x=4.0, we use the last segment (2,4) - (3,9) which has slope 5
        // So the result is 9 + (4-3)*5 = 9 + 5 = 14
        assert_relative_eq!(
            interp_extrapolate.evaluate(4.0).expect("Operation failed"),
            14.0
        );
    }

    #[test]
    fn test_convenience_functions() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0, 9.0];
        let xnew = array![0.5, 1.5, 2.5];

        // Test nearest interpolation
        let y_nearest =
            nearest_interpolate(&x.view(), &y.view(), &xnew.view()).expect("Operation failed");
        // Point 0.5 is exactly halfway between x[0]=0.0 and x[1]=1.0, so we default to the left point's value
        assert_relative_eq!(y_nearest[0], 0.0);
        // Point 1.5 is exactly halfway between x[1]=1.0 and x[2]=2.0, so we default to the left point's value
        assert_relative_eq!(y_nearest[1], 1.0);
        // Point 2.5 is exactly halfway between x[2]=2.0 and x[3]=3.0, so we default to the left point's value
        assert_relative_eq!(y_nearest[2], 4.0);

        // Test linear interpolation
        let y_linear =
            linear_interpolate(&x.view(), &y.view(), &xnew.view()).expect("Operation failed");
        assert_relative_eq!(y_linear[0], 0.5);
        assert_relative_eq!(y_linear[1], 2.5);
        assert_relative_eq!(y_linear[2], 6.5);

        // Test cubic interpolation (not-a-knot spline reproduces quadratics exactly)
        let x5 = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y5 = array![0.0, 1.0, 4.0, 9.0, 16.0];
        let y_cubic =
            cubic_interpolate(&x5.view(), &y5.view(), &xnew.view()).expect("Operation failed");
        assert_relative_eq!(y_cubic[0], 0.25, epsilon = 1e-10);
        assert_relative_eq!(y_cubic[1], 2.25, epsilon = 1e-10);
        assert_relative_eq!(y_cubic[2], 6.25, epsilon = 1e-10);

        // Test PCHIP interpolation
        let y_pchip =
            pchip_interpolate(&x.view(), &y.view(), &xnew.view(), false).expect("Operation failed");
        // For monotonically increasing data, PCHIP should preserve monotonicity
        assert!(y_pchip[0] > 0.0 && y_pchip[0] < 1.0);
        assert!(y_pchip[1] > 1.0 && y_pchip[1] < 4.0);
        assert!(y_pchip[2] > 4.0 && y_pchip[2] < 9.0);
    }

    #[test]
    fn test_error_conditions() {
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = array![0.0, 1.0, 4.0];

        // Test different lengths
        let result = Interp1d::new(
            &x.view(),
            &y.view(),
            InterpolationMethod::Linear,
            ExtrapolateMode::Error,
        );
        assert!(result.is_err());

        // Test unsorted x
        let x_unsorted = array![0.0, 2.0, 1.0, 3.0];
        let y_valid = array![0.0, 1.0, 4.0, 9.0];

        let result = Interp1d::new(
            &x_unsorted.view(),
            &y_valid.view(),
            InterpolationMethod::Linear,
            ExtrapolateMode::Error,
        );
        assert!(result.is_err());

        // Test too few points for cubic (1 point should fail, 2 points should succeed with linear fallback)
        let x_one = array![0.0];
        let y_one = array![0.0];

        let result = Interp1d::new(
            &x_one.view(),
            &y_one.view(),
            InterpolationMethod::Cubic,
            ExtrapolateMode::Error,
        );
        assert!(result.is_err());

        // 2 points should succeed (linear fallback)
        let x_short = array![0.0, 1.0];
        let y_short = array![0.0, 1.0];
        let result = Interp1d::new(
            &x_short.view(),
            &y_short.view(),
            InterpolationMethod::Cubic,
            ExtrapolateMode::Error,
        );
        assert!(result.is_ok());
        let interp = result.unwrap();
        assert_relative_eq!(interp.evaluate(0.5).unwrap(), 0.5, epsilon = 1e-12);
    }
}
