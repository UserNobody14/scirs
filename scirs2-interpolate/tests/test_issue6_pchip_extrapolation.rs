// Regression tests for Issue 6: PCHIP polynomial extrapolation
// PchipInterpolator and Interp1d(method=Pchip) should use Hermite
// polynomial continuation outside the data range, matching
// scipy.interpolate.PchipInterpolator(extrapolate=True).

use approx::{assert_abs_diff_eq, assert_relative_eq};
use scirs2_core::ndarray::{array, ArrayView1};
use scirs2_interpolate::interp1d::{ExtrapolateMode, Interp1d, InterpolationMethod};
use scirs2_interpolate::interp1d::pchip::PchipInterpolator;

#[test]
fn pchip_extrapolate_below_scipy() {
    let x = ArrayView1::from(&[0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0]);
    let y = ArrayView1::from(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 7.5, 9.0]);
    let interp =
        Interp1d::new(&x, &y, InterpolationMethod::Pchip, ExtrapolateMode::Extrapolate).unwrap();

    let result = interp.evaluate(-1.0).unwrap();
    // scipy PchipInterpolator(extrapolate=True)(-1.0) ≈ -3.0
    assert_abs_diff_eq!(result, -3.0, epsilon = 0.05);
}

#[test]
fn pchip_extrapolate_above_scipy() {
    let x = ArrayView1::from(&[0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0]);
    let y = ArrayView1::from(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 7.5, 9.0]);
    let interp =
        Interp1d::new(&x, &y, InterpolationMethod::Pchip, ExtrapolateMode::Extrapolate).unwrap();

    let result: f64 = interp.evaluate(11.0).unwrap();
    // Polynomial continuation should produce a finite value > 0
    assert!(result.is_finite());
    assert!(result > 0.0, "Expected positive extrapolation above, got {}", result);
}

#[test]
fn pchip_interpolator_polynomial_continuation() {
    let x = array![0.0, 1.0, 2.0, 3.0];
    let y = array![0.0, 1.0, 4.0, 9.0];
    let interp = PchipInterpolator::new(&x.view(), &y.view(), true).unwrap();

    let y_4: f64 = interp.evaluate(4.0).unwrap();
    assert!(y_4.is_finite());
    assert!(y_4 > 9.0, "Extrapolation at x=4 should be > 9.0, got {}", y_4);

    // At boundary should be exact
    let y_3: f64 = interp.evaluate(3.0).unwrap();
    assert_relative_eq!(y_3, 9.0, epsilon = 1e-10);

    // Far extrapolation may grow cubically -- this is correct
    let y_m1: f64 = interp.evaluate(-1.0).unwrap();
    assert!(y_m1.is_finite());
}

#[test]
fn pchip_in_bounds_unchanged() {
    let x = ArrayView1::from(&[0.0, 1.0, 2.0, 3.0]);
    let y = ArrayView1::from(&[0.0, 1.0, 4.0, 9.0]);
    let interp =
        Interp1d::new(&x, &y, InterpolationMethod::Pchip, ExtrapolateMode::Extrapolate).unwrap();

    // Knot values should be exact
    assert_abs_diff_eq!(interp.evaluate(0.0).unwrap(), 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(interp.evaluate(1.0).unwrap(), 1.0, epsilon = 1e-10);
    assert_abs_diff_eq!(interp.evaluate(2.0).unwrap(), 4.0, epsilon = 1e-10);
    assert_abs_diff_eq!(interp.evaluate(3.0).unwrap(), 9.0, epsilon = 1e-10);
}
