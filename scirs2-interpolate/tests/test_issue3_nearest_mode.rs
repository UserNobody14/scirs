// Regression tests for Issue 3: interpnd ExtrapolateMode::Nearest
// Clamp out-of-range coordinates to the grid boundary before interpolating.

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{Array, Array1, Array2, IxDyn};
use scirs2_interpolate::interpnd;

#[test]
fn rgi_nearest_clamp_above() {
    let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let values = Array::from_shape_vec(IxDyn(&[3]), vec![0.0, 10.0, 20.0]).unwrap();
    let interp = interpnd::RegularGridInterpolator::new(
        vec![x],
        values,
        interpnd::InterpolationMethod::Linear,
        interpnd::ExtrapolateMode::Nearest,
    )
    .unwrap();

    let xi = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
    let result = interp.__call__(&xi.view()).unwrap();
    assert_abs_diff_eq!(result[0], 20.0, epsilon = 1e-10);
}

#[test]
fn rgi_nearest_clamp_below() {
    let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let values = Array::from_shape_vec(IxDyn(&[3]), vec![0.0, 10.0, 20.0]).unwrap();
    let interp = interpnd::RegularGridInterpolator::new(
        vec![x],
        values,
        interpnd::InterpolationMethod::Linear,
        interpnd::ExtrapolateMode::Nearest,
    )
    .unwrap();

    let xi = Array2::from_shape_vec((1, 1), vec![-3.0]).unwrap();
    let result = interp.__call__(&xi.view()).unwrap();
    assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10);
}

#[test]
fn rgi_nearest_clamp_2d() {
    let x = Array1::from_vec(vec![0.0, 1.0]);
    let y = Array1::from_vec(vec![0.0, 1.0]);
    let mut values = Array::zeros(IxDyn(&[2, 2]));
    values[[0, 0].as_slice()] = 1.0;
    values[[0, 1].as_slice()] = 2.0;
    values[[1, 0].as_slice()] = 3.0;
    values[[1, 1].as_slice()] = 4.0;

    let interp = interpnd::RegularGridInterpolator::new(
        vec![x, y],
        values,
        interpnd::InterpolationMethod::Linear,
        interpnd::ExtrapolateMode::Nearest,
    )
    .unwrap();

    // Both dims out of range above: clamp to (1,1), value = 4.0
    let xi = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();
    let result = interp.__call__(&xi.view()).unwrap();
    assert_abs_diff_eq!(result[0], 4.0, epsilon = 1e-10);

    // Both dims out of range below: clamp to (0,0), value = 1.0
    let xi2 = Array2::from_shape_vec((1, 2), vec![-1.0, -1.0]).unwrap();
    let result2 = interp.__call__(&xi2.view()).unwrap();
    assert_abs_diff_eq!(result2[0], 1.0, epsilon = 1e-10);

    // In-bounds should still work
    let xi3 = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();
    let result3 = interp.__call__(&xi3.view()).unwrap();
    assert_abs_diff_eq!(result3[0], 2.5, epsilon = 1e-10);
}

#[test]
fn rgi_nearest_with_nearest_method() {
    let x = Array1::from_vec(vec![0.0, 1.0, 2.0]);
    let values = Array::from_shape_vec(IxDyn(&[3]), vec![0.0, 10.0, 20.0]).unwrap();
    let interp = interpnd::RegularGridInterpolator::new(
        vec![x],
        values,
        interpnd::InterpolationMethod::Nearest,
        interpnd::ExtrapolateMode::Nearest,
    )
    .unwrap();

    // Should clamp then pick nearest grid point
    let xi = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
    let result = interp.__call__(&xi.view()).unwrap();
    assert_abs_diff_eq!(result[0], 20.0, epsilon = 1e-10);
}
