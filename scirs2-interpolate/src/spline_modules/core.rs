//! Core cubic spline data structures and implementations
//!
//! This module contains the main `CubicSpline` struct and `CubicSplineBuilder` struct
//! along with their core implementations. These provide the fundamental data structures
//! for cubic spline interpolation.

use crate::error::{InterpolateError, InterpolateResult};
use crate::traits::InterpolationFloat;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use super::types::SplineBoundaryCondition;
use super::algorithms::*;

/// A cubic spline interpolator
///
/// This struct represents a constructed cubic spline that can be used for interpolation
/// and derivative computation. Each spline consists of piecewise cubic polynomials that
/// maintain C² continuity (continuous function, first, and second derivatives).
///
/// # Type Parameters
///
/// * `F` - The floating point type (f32 or f64)
///
/// # Structure
///
/// The spline stores:
/// - `x`: The sorted x coordinates of the data points
/// - `y`: The y coordinates of the data points
/// - `coeffs`: Polynomial coefficients for each segment
///
/// Each row of `coeffs` contains [a, b, c, d] representing the cubic polynomial:
/// ```text
/// y(x) = a + b*(x-x_i) + c*(x-x_i)² + d*(x-x_i)³
/// ```
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_interpolate::spline::CubicSpline;
///
/// let x = array![0.0, 1.0, 2.0, 3.0];
/// let y = array![0.0, 1.0, 4.0, 9.0];
///
/// let spline = CubicSpline::new(&x.view(), &y.view()).expect("Operation failed");
/// let value = spline.evaluate(1.5).expect("Operation failed");
/// ```
#[derive(Debug, Clone)]
pub struct CubicSpline<F: InterpolationFloat> {
    /// X coordinates (must be sorted)
    x: Array1<F>,
    /// Y coordinates
    y: Array1<F>,
    /// Coefficients for cubic polynomials (n-1 segments, 4 coefficients each)
    /// Each row represents [a, b, c, d] for a segment
    /// y(x) = a + b*(x-x_i) + c*(x-x_i)^2 + d*(x-x_i)^3
    coeffs: Array2<F>,
}

/// Builder for cubic splines with custom boundary conditions
///
/// This builder allows for flexible construction of cubic splines with different
/// boundary conditions. It uses the builder pattern to provide a fluent API
/// for spline configuration.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_interpolate::spline::{CubicSpline, SplineBoundaryCondition};
///
/// let x = array![0.0, 1.0, 2.0, 3.0];
/// let y = array![0.0, 1.0, 4.0, 9.0];
///
/// let spline = CubicSpline::builder()
///     .x(x)
///     .y(y)
///     .boundary_condition(SplineBoundaryCondition::NotAKnot)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CubicSplineBuilder<F: InterpolationFloat> {
    x: Option<Array1<F>>,
    y: Option<Array1<F>>,
    boundary_condition: SplineBoundaryCondition<F>,
}

impl<F: InterpolationFloat> CubicSplineBuilder<F> {
    /// Create a new builder with default settings
    ///
    /// Default boundary condition is Natural (zero second derivative at endpoints).
    pub fn new() -> Self {
        Self {
            x: None,
            y: None,
            boundary_condition: SplineBoundaryCondition::Natural,
        }
    }

    /// Set the x coordinates
    ///
    /// # Arguments
    ///
    /// * `x` - Array of x coordinates (must be sorted in ascending order)
    ///
    /// # Returns
    ///
    /// The builder with x coordinates set
    pub fn x(mut self, x: Array1<F>) -> Self {
        self.x = Some(x);
        self
    }

    /// Set the y coordinates
    ///
    /// # Arguments
    ///
    /// * `y` - Array of y coordinates (must have same length as x)
    ///
    /// # Returns
    ///
    /// The builder with y coordinates set
    pub fn y(mut self, y: Array1<F>) -> Self {
        self.y = Some(y);
        self
    }

    /// Set the boundary condition
    ///
    /// # Arguments
    ///
    /// * `bc` - The boundary condition to use
    ///
    /// # Returns
    ///
    /// The builder with boundary condition set
    pub fn boundary_condition(mut self, bc: SplineBoundaryCondition<F>) -> Self {
        self.boundary_condition = bc;
        self
    }

    /// Build the spline
    ///
    /// # Returns
    ///
    /// A new `CubicSpline` object or an error if construction fails
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - x or y coordinates are not set
    /// - Arrays have different lengths
    /// - Insufficient points for the boundary condition
    /// - x coordinates are not sorted
    /// - Numerical issues during construction
    pub fn build(self) -> InterpolateResult<CubicSpline<F>> {
        let x = self
            .x
            .ok_or_else(|| InterpolateError::invalid_input("x coordinates not set".to_string()))?;
        let y = self
            .y
            .ok_or_else(|| InterpolateError::invalid_input("y coordinates not set".to_string()))?;

        CubicSpline::with_boundary_condition(&x.view(), &y.view(), self.boundary_condition)
    }
}

impl<F: InterpolationFloat> Default for CubicSplineBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: InterpolationFloat + ToString> CubicSpline<F> {
    /// Create a new builder for cubic splines
    ///
    /// # Returns
    ///
    /// A new `CubicSplineBuilder` instance
    pub fn builder() -> CubicSplineBuilder<F> {
        CubicSplineBuilder::new()
    }

    /// Create a new cubic spline with natural boundary conditions
    ///
    /// Natural boundary conditions set the second derivative to zero at both endpoints,
    /// which minimizes the total curvature of the spline.
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    ///
    /// # Returns
    ///
    /// A new `CubicSpline` object
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Arrays have different lengths
    /// - Less than 3 points provided
    /// - x coordinates are not sorted
    /// - Numerical issues during construction
    ///
    /// # Examples
    ///
    /// ```rust
    /// use scirs2_core::ndarray::array;
    /// use scirs2_interpolate::spline::CubicSpline;
    ///
    /// let x = array![0.0, 1.0, 2.0, 3.0];
    /// let y = array![0.0, 1.0, 4.0, 9.0];
    ///
    /// let spline = CubicSpline::new(&x.view(), &y.view()).expect("Operation failed");
    ///
    /// // Interpolate at x = 1.5
    /// let y_interp = spline.evaluate(1.5).expect("Operation failed");
    /// println!("Interpolated value at x=1.5: {}", y_interp);
    /// ```
    pub fn new(x: &ArrayView1<F>, y: &ArrayView1<F>) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 3 {
            return Err(InterpolateError::insufficient_points(
                3,
                x.len(),
                "cubic spline",
            ));
        }

        // Check that x is sorted
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x values must be sorted in ascending order".to_string(),
                ));
            }
        }

        // Get coefficients for natural cubic spline
        let coeffs = compute_natural_cubic_spline(x, y)?;

        Ok(CubicSpline {
            x: x.to_owned(),
            y: y.to_owned(),
            coeffs,
        })
    }

    /// Get the x coordinates
    ///
    /// # Returns
    ///
    /// Reference to the array of x coordinates
    pub fn x(&self) -> &Array1<F> {
        &self.x
    }

    /// Get the y coordinates
    ///
    /// # Returns
    ///
    /// Reference to the array of y coordinates
    pub fn y(&self) -> &Array1<F> {
        &self.y
    }

    /// Get the polynomial coefficients
    ///
    /// # Returns
    ///
    /// Reference to the 2D array of polynomial coefficients
    /// Shape: (n-1, 4) where n is the number of data points
    /// Each row contains [a, b, c, d] for the polynomial in that segment
    pub fn coeffs(&self) -> &Array2<F> {
        &self.coeffs
    }

    /// Create a new cubic spline with not-a-knot boundary conditions
    ///
    /// Not-a-knot boundary conditions force the third derivative to be continuous
    /// at the second and second-to-last data points, providing maximum smoothness.
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    ///
    /// # Returns
    ///
    /// A new `CubicSpline` object
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Arrays have different lengths
    /// - Less than 4 points provided (required for not-a-knot)
    /// - x coordinates are not sorted
    /// - Numerical issues during construction
    pub fn new_not_a_knot(x: &ArrayView1<F>, y: &ArrayView1<F>) -> InterpolateResult<Self> {
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        if x.len() < 2 {
            return Err(InterpolateError::insufficient_points(
                2,
                x.len(),
                "not-a-knot cubic spline",
            ));
        }

        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x values must be sorted in ascending order".to_string(),
                ));
            }
        }

        let n = x.len();

        if n == 2 {
            // Linear fallback: y = y0 + slope*(x - x0), stored as cubic with c=d=0
            let h = x[1] - x[0];
            let slope = (y[1] - y[0]) / h;
            let mut coeffs = Array2::<F>::zeros((1, 4));
            coeffs[[0, 0]] = y[0];
            coeffs[[0, 1]] = slope;
            return Ok(CubicSpline {
                x: x.to_owned(),
                y: y.to_owned(),
                coeffs,
            });
        }

        if n == 3 {
            // Quadratic (parabolic) fit through 3 points, matching scipy's behavior.
            // Fit y = a + b*(x-x0) + c*(x-x0)^2 using Lagrange interpolation,
            // then express each segment as a cubic with d=0.
            let h0 = x[1] - x[0];
            let h1 = x[2] - x[1];
            let d0 = (y[1] - y[0]) / h0;
            let d1 = (y[2] - y[1]) / h1;
            let c_val = (d1 - d0) / (x[2] - x[0]);

            let mut coeffs = Array2::<F>::zeros((2, 4));
            // Segment 0: [x0, x1]
            coeffs[[0, 0]] = y[0];
            coeffs[[0, 1]] = d0 - c_val * h0; // first derivative at x0
            // Actually for the quadratic a + b*(x-x0) + c*(x-x0)^2:
            // y(x0) = a = y[0], y(x1) = a + b*h0 + c*h0^2 = y[1], y(x2) = a + b*(x2-x0) + c*(x2-x0)^2 = y[2]
            // b = d0 - c*h0
            // But we need segment-local coefficients.
            let b0 = d0 - c_val * h0;
            coeffs[[0, 0]] = y[0];
            coeffs[[0, 1]] = b0;
            coeffs[[0, 2]] = c_val;
            coeffs[[0, 3]] = F::zero();

            // Segment 1: [x1, x2], re-expand about x1
            // y(x) = y0 + b0*(x-x0) + c*(x-x0)^2
            // At x1: y'(x1) = b0 + 2*c*h0
            let b1 = b0 + F::from_f64(2.0).unwrap() * c_val * h0;
            coeffs[[1, 0]] = y[1];
            coeffs[[1, 1]] = b1;
            coeffs[[1, 2]] = c_val;
            coeffs[[1, 3]] = F::zero();

            return Ok(CubicSpline {
                x: x.to_owned(),
                y: y.to_owned(),
                coeffs,
            });
        }

        let coeffs = compute_not_a_knot_cubic_spline(x, y)?;

        Ok(CubicSpline {
            x: x.to_owned(),
            y: y.to_owned(),
            coeffs,
        })
    }

    /// Create a cubic spline with custom boundary conditions
    ///
    /// This is the most general constructor that supports all boundary condition types.
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinates (must be sorted in ascending order)
    /// * `y` - The y coordinates (must have the same length as x)
    /// * `bc` - The boundary condition to use
    ///
    /// # Returns
    ///
    /// A new `CubicSpline` object
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Arrays have different lengths
    /// - Insufficient points for the chosen boundary condition
    /// - x coordinates are not sorted
    /// - Numerical issues during construction
    /// - Invalid boundary condition parameters
    pub fn with_boundary_condition(
        x: &ArrayView1<F>,
        y: &ArrayView1<F>,
        bc: SplineBoundaryCondition<F>,
    ) -> InterpolateResult<Self> {
        // Check inputs
        if x.len() != y.len() {
            return Err(InterpolateError::invalid_input(
                "x and y arrays must have the same length".to_string(),
            ));
        }

        // Check minimum points based on boundary condition
        let min_points = match bc {
            SplineBoundaryCondition::NotAKnot => 4,
            _ => 3,
        };

        if x.len() < min_points {
            return Err(InterpolateError::insufficient_points(
                min_points,
                x.len(),
                &format!("cubic spline with {:?} boundary condition", bc),
            ));
        }

        // Check that x is sorted
        for i in 1..x.len() {
            if x[i] <= x[i - 1] {
                return Err(InterpolateError::invalid_input(
                    "x values must be sorted in ascending order".to_string(),
                ));
            }
        }

        // Validate periodic boundary condition
        if let SplineBoundaryCondition::Periodic = bc {
            let tolerance = F::from_f64(1e-10).unwrap_or_else(|| F::epsilon());
            if (y[0] - y[y.len() - 1]).abs() > tolerance {
                return Err(InterpolateError::invalid_input(
                    "For periodic boundary conditions, first and last y values must be equal".to_string(),
                ));
            }
        }

        // Get coefficients based on boundary condition
        let coeffs = match bc {
            SplineBoundaryCondition::Natural => compute_natural_cubic_spline(x, y)?,
            SplineBoundaryCondition::NotAKnot => compute_not_a_knot_cubic_spline(x, y)?,
            SplineBoundaryCondition::Clamped(dy0, dyn_) => {
                compute_clamped_cubic_spline(x, y, dy0, dyn_)?
            }
            SplineBoundaryCondition::Periodic => compute_periodic_cubic_spline(x, y)?,
            SplineBoundaryCondition::SecondDerivative(d2y0, d2yn) => {
                compute_second_derivative_cubic_spline(x, y, d2y0, d2yn)?
            }
            SplineBoundaryCondition::ParabolicRunout => {
                compute_parabolic_runout_cubic_spline(x, y)?
            }
        };

        Ok(CubicSpline {
            x: x.to_owned(),
            y: y.to_owned(),
            coeffs,
        })
    }
}