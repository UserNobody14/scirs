// Regression tests for Issue 2: RegularGridInterpolator single-point axes
// A single-point axis should be valid -- interpolation along that dimension
// is trivially the constant value. scipy handles this gracefully.

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{Array, Array1, Array2, IxDyn};
use scirs2_interpolate::interpnd;

#[test]
fn rgi_single_point_2d_grid() {
    // One axis has 1 point, the other has 2
    let x = Array1::from_vec(vec![5.0]);
    let y = Array1::from_vec(vec![0.0, 1.0]);
    let mut values = Array::zeros(IxDyn(&[1, 2]));
    values[[0, 0].as_slice()] = 42.0;
    values[[0, 1].as_slice()] = 100.0;

    let interp = interpnd::RegularGridInterpolator::new(
        vec![x, y],
        values,
        interpnd::InterpolationMethod::Linear,
        interpnd::ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    // scipy RGI: 71.0
    let xi = Array2::from_shape_vec((1, 2), vec![5.0, 0.5]).unwrap();
    let result = interp.__call__(&xi.view()).unwrap();
    assert_abs_diff_eq!(result[0], 71.0, epsilon = 1e-10);

    // Exact corners
    let xi2 = Array2::from_shape_vec((2, 2), vec![5.0, 0.0, 5.0, 1.0]).unwrap();
    let result2 = interp.__call__(&xi2.view()).unwrap();
    assert_abs_diff_eq!(result2[0], 42.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result2[1], 100.0, epsilon = 1e-10);
}

#[test]
fn rgi_single_point_1d() {
    let x = Array1::from_vec(vec![5.0]);
    let values = Array::from_shape_vec(IxDyn(&[1]), vec![42.0]).unwrap();
    let interp = interpnd::RegularGridInterpolator::new(
        vec![x],
        values,
        interpnd::InterpolationMethod::Linear,
        interpnd::ExtrapolateMode::Extrapolate,
    )
    .unwrap();

    let xi = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
    let result = interp.__call__(&xi.view()).unwrap();
    assert_abs_diff_eq!(result[0], 42.0, epsilon = 1e-10);
}

#[test]
fn rgi_rejects_empty_axis() {
    let x = Array1::<f64>::zeros(0);
    let values = Array::zeros(IxDyn(&[0]));
    let result = interpnd::RegularGridInterpolator::new(
        vec![x],
        values,
        interpnd::InterpolationMethod::Linear,
        interpnd::ExtrapolateMode::Extrapolate,
    );
    assert!(result.is_err());
}
