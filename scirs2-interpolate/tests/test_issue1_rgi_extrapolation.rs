// Regression tests for Issue 1: RegularGridInterpolator linear extrapolation
// ExtrapolateMode::Extrapolate with Linear should continue the boundary slope,
// matching scipy.RegularGridInterpolator(bounds_error=False, fill_value=None).

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{Array, Array1, Array2, IxDyn};
use scirs2_interpolate::interpnd;

#[test]
fn rgi_linear_extrapolate_above_1d() {
    let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let values = Array::from_shape_vec(IxDyn(&[3]), vec![0.0, 10.0, 20.0]).unwrap();
    let interp = interpnd::RegularGridInterpolator::new(
        vec![x],
        values,
        interpnd::InterpolationMethod::Linear,
        interpnd::ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    let xi = Array2::from_shape_vec((1, 1), vec![3.0]).unwrap();
    let result = interp.__call__(&xi.view()).unwrap();
    // scipy: 30.0 (slope=10, 20 + 10*(3-2))
    assert_abs_diff_eq!(result[0], 30.0, epsilon = 1e-10);
}

#[test]
fn rgi_linear_extrapolate_below_1d() {
    let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let values = Array::from_shape_vec(IxDyn(&[3]), vec![0.0, 10.0, 20.0]).unwrap();
    let interp = interpnd::RegularGridInterpolator::new(
        vec![x],
        values,
        interpnd::InterpolationMethod::Linear,
        interpnd::ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    let xi = Array2::from_shape_vec((1, 1), vec![-1.0]).unwrap();
    let result = interp.__call__(&xi.view()).unwrap();
    // scipy: -10.0
    assert_abs_diff_eq!(result[0], -10.0, epsilon = 1e-10);
}

#[test]
fn rgi_linear_extrapolate_2d() {
    let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let y = Array1::from_vec(vec![0.0, 1.0]);
    let mut values = Array::zeros(IxDyn(&[3, 2]));
    for i in 0..3 {
        for j in 0..2 {
            values[[i, j].as_slice()] = (i * 10 + j * 5) as f64;
        }
    }

    let interp = interpnd::RegularGridInterpolator::new(
        vec![x, y],
        values,
        interpnd::InterpolationMethod::Linear,
        interpnd::ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // Extrapolate beyond x=2 at y=0.5
    let xi = Array2::from_shape_vec((1, 2), vec![3.0, 0.5]).unwrap();
    let result = interp.__call__(&xi.view()).unwrap();
    assert_abs_diff_eq!(result[0], 32.5, epsilon = 1e-10);
}

#[test]
fn rgi_in_bounds_unchanged() {
    let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let values = Array::from_shape_vec(IxDyn(&[3]), vec![0.0, 10.0, 20.0]).unwrap();
    let interp = interpnd::RegularGridInterpolator::new(
        vec![x],
        values,
        interpnd::InterpolationMethod::Linear,
        interpnd::ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // Grid points
    let xi = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
    let result = interp.__call__(&xi.view()).unwrap();
    assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result[1], 10.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result[2], 20.0, epsilon = 1e-10);

    // Midpoint
    let xi2 = Array2::from_shape_vec((1, 1), vec![1.5]).unwrap();
    let result2 = interp.__call__(&xi2.view()).unwrap();
    assert_abs_diff_eq!(result2[0], 15.0, epsilon = 1e-10);
}
