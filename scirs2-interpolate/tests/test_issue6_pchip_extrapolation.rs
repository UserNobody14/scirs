// Regression tests for Issue 6: PCHIP extrapolation modes
//
// PchipInterpolator supports two extrapolation strategies:
//   - Linear (default): stable linear extension using endpoint derivatives
//   - Polynomial: Hermite cubic continuation matching scipy PchipInterpolator
//
// Interp1d(method=Pchip, ExtrapolateMode::Extrapolate) uses Polynomial to
// match scipy behavior.

use approx::{assert_abs_diff_eq, assert_relative_eq};
use scirs2_core::ndarray::{array, ArrayView1};
use scirs2_interpolate::interp1d::{
    ExtrapolateMode, Interp1d, InterpolationMethod, PchipExtrapolateMode, PchipInterpolator,
};

// -- Polynomial continuation via Interp1d (scipy-compatible path) --

#[test]
fn interp1d_pchip_extrapolate_below() {
    let x = ArrayView1::from(&[0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0]);
    let y = ArrayView1::from(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 7.5, 9.0]);
    let interp =
        Interp1d::new(&x, &y, InterpolationMethod::Pchip, ExtrapolateMode::Extrapolate).unwrap();

    let result = interp.evaluate(-1.0).unwrap();
    // scipy PchipInterpolator(extrapolate=True)(-1.0) ≈ -3.0
    assert_abs_diff_eq!(result, -3.0, epsilon = 0.05);
}

#[test]
fn interp1d_pchip_extrapolate_above() {
    let x = ArrayView1::from(&[0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0]);
    let y = ArrayView1::from(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 7.5, 9.0]);
    let interp =
        Interp1d::new(&x, &y, InterpolationMethod::Pchip, ExtrapolateMode::Extrapolate).unwrap();

    let result: f64 = interp.evaluate(11.0).unwrap();
    assert!(result.is_finite());
    assert!(result > 0.0, "Expected positive extrapolation above, got {}", result);
}

#[test]
fn interp1d_pchip_in_bounds_unchanged() {
    let x = ArrayView1::from(&[0.0, 1.0, 2.0, 3.0]);
    let y = ArrayView1::from(&[0.0, 1.0, 4.0, 9.0]);
    let interp =
        Interp1d::new(&x, &y, InterpolationMethod::Pchip, ExtrapolateMode::Extrapolate).unwrap();

    assert_abs_diff_eq!(interp.evaluate(0.0).unwrap(), 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(interp.evaluate(1.0).unwrap(), 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(interp.evaluate(2.0).unwrap(), 4.0, epsilon = 1e-10);
    assert_abs_diff_eq!(interp.evaluate(3.0).unwrap(), 9.0, epsilon = 1e-10);
}

// -- PchipInterpolator polynomial mode (explicit opt-in) --

#[test]
fn pchip_polynomial_continuation() {
    let x = array![0.0, 1.0, 2.0, 3.0];
    let y = array![0.0, 1.0, 4.0, 9.0];
    let interp = PchipInterpolator::new(&x.view(), &y.view(), true)
        .unwrap()
        .with_extrapolate_mode(PchipExtrapolateMode::Polynomial);

    let y_4: f64 = interp.evaluate(4.0).unwrap();
    assert!(y_4.is_finite());
    assert!(y_4 > 9.0, "Extrapolation at x=4 should be > 9.0, got {}", y_4);

    let y_3: f64 = interp.evaluate(3.0).unwrap();
    assert_relative_eq!(y_3, 9.0, epsilon = 1e-10);

    let y_m1: f64 = interp.evaluate(-1.0).unwrap();
    assert!(y_m1.is_finite());
}

// -- PchipInterpolator linear mode (default, stable) --

#[test]
fn pchip_linear_extension_is_default() {
    let x = array![0.0, 1.0, 2.0, 3.0];
    let y = array![0.0, 1.0, 4.0, 9.0];
    let interp = PchipInterpolator::new(&x.view(), &y.view(), true).unwrap();

    let y_4: f64 = interp.evaluate(4.0).unwrap();
    let y_5: f64 = interp.evaluate(5.0).unwrap();
    let y_6: f64 = interp.evaluate(6.0).unwrap();

    // Linear extension: slope between consecutive extrapolated points is constant
    let slope1 = y_5 - y_4;
    let slope2 = y_6 - y_5;
    assert_relative_eq!(slope1, slope2, epsilon = 1e-10);

    // Far extrapolation stays moderate (linear, not cubic blowup)
    let y_50: f64 = interp.evaluate(50.0).unwrap();
    assert!(y_50.is_finite());
    assert!(y_50.abs() < 1000.0, "Linear extrapolation should be bounded, got {}", y_50);
}
