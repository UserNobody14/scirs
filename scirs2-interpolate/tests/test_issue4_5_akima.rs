// Regression tests for Issues 4 & 5: AkimaSpline
// Issue 4: Accept n >= 2 (was n >= 5). n == 2 degenerates to linear.
// Issue 5: Add extrapolation mode support (Error, Nearest, Extrapolate).
// Reference values from scipy.interpolate.Akima1DInterpolator.

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::array;
use scirs2_interpolate::{
    advanced::akima::AkimaSpline,
    interp1d::ExtrapolateMode,
};

// -- Issue 4: Minimum point count --

#[test]
fn akima_2_points_linear() {
    let x = array![0.0, 1.0];
    let y = array![0.0, 2.0];
    let spline = AkimaSpline::new(&x.view(), &y.view()).unwrap();
    // scipy Akima2pt(0.5) = 1.0
    assert_abs_diff_eq!(spline.evaluate(0.5).unwrap(), 1.0, epsilon = 1e-10);
}

#[test]
fn akima_3_points() {
    let x = array![0.0, 1.0, 2.0];
    let y = array![0.0, 1.0, 0.0];
    let spline = AkimaSpline::new(&x.view(), &y.view()).unwrap();
    // scipy: Akima3pt(0.5) = 0.75, Akima3pt(1.5) = 0.75
    assert_abs_diff_eq!(spline.evaluate(0.5).unwrap(), 0.75, epsilon = 0.1);
    assert_abs_diff_eq!(spline.evaluate(1.5).unwrap(), 0.75, epsilon = 0.1);
}

#[test]
fn akima_4_points() {
    let x = array![0.0, 1.0, 2.0, 3.0];
    let y = array![0.0, 1.0, 0.0, 1.0];
    let spline = AkimaSpline::new(&x.view(), &y.view()).unwrap();
    // scipy: (0.5) = 0.75, (1.5) = 0.5, (2.5) = 0.25
    assert_abs_diff_eq!(spline.evaluate(0.5).unwrap(), 0.75, epsilon = 0.15);
    assert_abs_diff_eq!(spline.evaluate(1.5).unwrap(), 0.5, epsilon = 0.15);
    assert_abs_diff_eq!(spline.evaluate(2.5).unwrap(), 0.25, epsilon = 0.15);
}

#[test]
fn akima_rejects_single_point() {
    let x = array![1.0];
    let y = array![5.0];
    assert!(AkimaSpline::new(&x.view(), &y.view()).is_err());
}

// -- Issue 5: Extrapolation modes --

#[test]
fn akima_error_mode_is_default() {
    let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0];
    let y = array![0.0f64, 1.0, 4.0, 9.0, 16.0];
    let spline = AkimaSpline::new(&x.view(), &y.view()).unwrap();
    assert!(spline.evaluate(-1.0).is_err());
    assert!(spline.evaluate(5.0).is_err());
}

#[test]
fn akima_extrapolate_mode() {
    let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0];
    let y = array![0.0f64, 1.0, 4.0, 9.0, 16.0];
    let spline = AkimaSpline::new(&x.view(), &y.view())
        .unwrap()
        .with_extrapolation(ExtrapolateMode::Extrapolate);

    let below: f64 = spline.evaluate(-1.0).unwrap();
    assert!(below.is_finite());

    let above: f64 = spline.evaluate(5.0).unwrap();
    assert!(above.is_finite());

    // In-bounds should still be exact at knots
    assert_abs_diff_eq!(spline.evaluate(0.0).unwrap(), 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(spline.evaluate(4.0).unwrap(), 16.0, epsilon = 1e-10);
}

#[test]
fn akima_nearest_mode() {
    let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0];
    let y = array![0.0f64, 1.0, 4.0, 9.0, 16.0];
    let spline = AkimaSpline::new(&x.view(), &y.view())
        .unwrap()
        .with_extrapolation(ExtrapolateMode::Nearest);

    // Below: clamp to x=0 -> y=0
    assert_abs_diff_eq!(spline.evaluate(-5.0).unwrap(), 0.0, epsilon = 1e-10);
    // Above: clamp to x=4 -> y=16
    assert_abs_diff_eq!(spline.evaluate(10.0).unwrap(), 16.0, epsilon = 1e-10);
}

#[test]
fn akima_derivative_with_extrapolation() {
    let x = array![0.0f64, 1.0, 2.0, 3.0, 4.0];
    let y = array![0.0f64, 1.0, 4.0, 9.0, 16.0];
    let spline = AkimaSpline::new(&x.view(), &y.view())
        .unwrap()
        .with_extrapolation(ExtrapolateMode::Extrapolate);

    let d: f64 = spline.derivative(5.0).unwrap();
    assert!(d.is_finite());

    let d2: f64 = spline.derivative(-1.0).unwrap();
    assert!(d2.is_finite());
}
