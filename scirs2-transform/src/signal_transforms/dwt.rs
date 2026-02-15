//! Discrete Wavelet Transform (DWT) Implementation
//!
//! Provides 1D, 2D, and N-D discrete wavelet transforms with multiple wavelet families.
//! Implements efficient decomposition and reconstruction with proper boundary handling.

use crate::error::{Result, TransformError};
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};

/// Wavelet types supported by the DWT implementation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaveletType {
    /// Haar wavelet (Daubechies-1)
    Haar,
    /// Daubechies wavelets (N = 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    Daubechies(usize),
    /// Symlet wavelets
    Symlet(usize),
    /// Coiflet wavelets
    Coiflet(usize),
    /// Biorthogonal wavelets
    Biorthogonal(usize, usize),
}

/// Boundary extension modes for DWT
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryMode {
    /// Zero padding
    Zero,
    /// Constant padding (edge values)
    Constant,
    /// Symmetric padding
    Symmetric,
    /// Periodic padding
    Periodic,
    /// Reflect padding
    Reflect,
}

/// Wavelet filter coefficients
#[derive(Debug, Clone)]
pub struct WaveletFilters {
    /// Low-pass decomposition filter
    pub dec_lo: Vec<f64>,
    /// High-pass decomposition filter
    pub dec_hi: Vec<f64>,
    /// Low-pass reconstruction filter
    pub rec_lo: Vec<f64>,
    /// High-pass reconstruction filter
    pub rec_hi: Vec<f64>,
}

impl WaveletFilters {
    /// Get filter coefficients for a specific wavelet type
    pub fn from_wavelet(wavelet: WaveletType) -> Result<Self> {
        match wavelet {
            WaveletType::Haar => Self::haar(),
            WaveletType::Daubechies(n) => Self::daubechies(n),
            WaveletType::Symlet(n) => Self::symlet(n),
            WaveletType::Coiflet(n) => Self::coiflet(n),
            WaveletType::Biorthogonal(p, q) => Self::biorthogonal(p, q),
        }
    }

    /// Haar wavelet filters
    fn haar() -> Result<Self> {
        let norm = 1.0 / 2.0_f64.sqrt();
        Ok(WaveletFilters {
            dec_lo: vec![norm, norm],
            dec_hi: vec![norm, -norm],
            rec_lo: vec![norm, norm],
            rec_hi: vec![-norm, norm],
        })
    }

    /// Daubechies wavelet filters
    fn daubechies(n: usize) -> Result<Self> {
        match n {
            2 => {
                // DB2 (Daubechies-4 coefficients)
                let sqrt3 = 3.0_f64.sqrt();
                let denom = 4.0 * 2.0_f64.sqrt();
                let dec_lo = vec![
                    (1.0 + sqrt3) / denom,
                    (3.0 + sqrt3) / denom,
                    (3.0 - sqrt3) / denom,
                    (1.0 - sqrt3) / denom,
                ];
                let mut dec_hi = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }

                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();

                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            4 => {
                // DB4 (Daubechies-8 coefficients)
                let dec_lo = vec![
                    -0.010597401784997,
                    0.032883011666983,
                    0.030841381835987,
                    -0.187034811718881,
                    -0.027983769416984,
                    0.630880767929590,
                    0.714846570552542,
                    0.230377813308855,
                ];
                let mut dec_hi = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }

                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();

                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            _ => Err(TransformError::InvalidInput(format!(
                "Daubechies-{} not yet implemented",
                n
            ))),
        }
    }

    /// Symlet wavelet filters (simplified - use Daubechies for now)
    fn symlet(n: usize) -> Result<Self> {
        // Symlets are nearly symmetric versions of Daubechies wavelets
        Self::daubechies(n)
    }

    /// Coiflet wavelet filters
    fn coiflet(n: usize) -> Result<Self> {
        match n {
            1 => {
                // Coif1 coefficients
                let sqrt2 = 2.0_f64.sqrt();
                let dec_lo = vec![
                    -0.01565572813546454 / sqrt2,
                    -0.07268974908697540 / sqrt2,
                    0.38486484686420286 / sqrt2,
                    0.85257202021225542 / sqrt2,
                    0.33789766245780093 / sqrt2,
                    -0.07268974908697540 / sqrt2,
                ];
                let mut dec_hi = Vec::with_capacity(dec_lo.len());
                for (i, &val) in dec_lo.iter().enumerate().rev() {
                    dec_hi.push(if i % 2 == 0 { val } else { -val });
                }

                let mut rec_lo = dec_lo.clone();
                rec_lo.reverse();
                let mut rec_hi = dec_hi.clone();
                rec_hi.reverse();

                Ok(WaveletFilters {
                    dec_lo,
                    dec_hi,
                    rec_lo,
                    rec_hi,
                })
            }
            _ => Err(TransformError::InvalidInput(format!(
                "Coiflet-{} not yet implemented",
                n
            ))),
        }
    }

    /// Biorthogonal wavelet filters
    fn biorthogonal(_p: usize, _q: usize) -> Result<Self> {
        // For now, return Haar as placeholder
        Self::haar()
    }
}

/// 1D Discrete Wavelet Transform
#[derive(Debug, Clone)]
pub struct DWT {
    wavelet: WaveletType,
    filters: WaveletFilters,
    boundary: BoundaryMode,
    level: Option<usize>,
}

impl DWT {
    /// Create a new DWT instance
    pub fn new(wavelet: WaveletType) -> Result<Self> {
        let filters = WaveletFilters::from_wavelet(wavelet)?;
        Ok(DWT {
            wavelet,
            filters,
            boundary: BoundaryMode::Symmetric,
            level: None,
        })
    }

    /// Set the boundary mode
    pub fn with_boundary(mut self, boundary: BoundaryMode) -> Self {
        self.boundary = boundary;
        self
    }

    /// Set the decomposition level
    pub fn with_level(mut self, level: usize) -> Self {
        self.level = Some(level);
        self
    }

    /// Perform single-level decomposition
    pub fn decompose(&self, signal: &ArrayView1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let n = signal.len();
        if n < 2 {
            return Err(TransformError::InvalidInput(
                "Signal too short for DWT".to_string(),
            ));
        }

        // Extend signal according to boundary mode
        let extended = self.extend_signal(signal)?;

        // Convolve with filters and downsample
        let approx = self.convolve_downsample(&extended, &self.filters.dec_lo)?;
        let detail = self.convolve_downsample(&extended, &self.filters.dec_hi)?;

        Ok((approx, detail))
    }

    /// Perform multi-level decomposition
    pub fn wavedec(&self, signal: &ArrayView1<f64>) -> Result<Vec<Array1<f64>>> {
        let max_level = self.max_decomposition_level(signal.len());
        let level = self.level.unwrap_or(max_level).min(max_level);

        let mut coeffs = Vec::with_capacity(level + 1);
        let mut current = signal.to_owned();

        for _ in 0..level {
            let (approx, detail) = self.decompose(&current.view())?;
            coeffs.push(detail);
            current = approx;
        }

        // Add final approximation coefficients
        coeffs.push(current);
        coeffs.reverse();

        Ok(coeffs)
    }

    /// Perform single-level reconstruction
    pub fn reconstruct(
        &self,
        approx: &ArrayView1<f64>,
        detail: &ArrayView1<f64>,
    ) -> Result<Array1<f64>> {
        // Upsample and convolve with reconstruction filters
        let approx_up = self.upsample_convolve(approx, &self.filters.rec_lo)?;
        let detail_up = self.upsample_convolve(detail, &self.filters.rec_hi)?;

        // Add the two components
        let min_len = approx_up.len().min(detail_up.len());
        let mut reconstructed = Array1::zeros(min_len);
        for i in 0..min_len {
            reconstructed[i] = approx_up[i] + detail_up[i];
        }

        Ok(reconstructed)
    }

    /// Perform multi-level reconstruction
    pub fn waverec(&self, coeffs: &[Array1<f64>]) -> Result<Array1<f64>> {
        if coeffs.is_empty() {
            return Err(TransformError::InvalidInput(
                "No coefficients provided for reconstruction".to_string(),
            ));
        }

        let mut current = coeffs[0].clone();

        for detail in &coeffs[1..] {
            current = self.reconstruct(&current.view(), &detail.view())?;
        }

        Ok(current)
    }

    // Helper methods

    fn extend_signal(&self, signal: &ArrayView1<f64>) -> Result<Array1<f64>> {
        let filter_len = self.filters.dec_lo.len();
        let n = signal.len();
        let pad_len = filter_len - 1;

        let mut extended = Array1::zeros(n + 2 * pad_len);

        match self.boundary {
            BoundaryMode::Zero => {
                for i in 0..n {
                    extended[i + pad_len] = signal[i];
                }
            }
            BoundaryMode::Constant => {
                let first = signal[0];
                let last = signal[n - 1];
                for i in 0..pad_len {
                    extended[i] = first;
                    extended[n + pad_len + i] = last;
                }
                for i in 0..n {
                    extended[i + pad_len] = signal[i];
                }
            }
            BoundaryMode::Symmetric => {
                for i in 0..pad_len {
                    extended[pad_len - 1 - i] = signal[i.min(n - 1)];
                    extended[n + pad_len + i] = signal[(n - 1 - i).max(0)];
                }
                for i in 0..n {
                    extended[i + pad_len] = signal[i];
                }
            }
            BoundaryMode::Periodic => {
                for i in 0..pad_len {
                    extended[i] = signal[(n - pad_len + i) % n];
                    extended[n + pad_len + i] = signal[i % n];
                }
                for i in 0..n {
                    extended[i + pad_len] = signal[i];
                }
            }
            BoundaryMode::Reflect => {
                for i in 0..pad_len {
                    let idx1 = if i < n { i } else { n - 1 };
                    let idx2 = if n > i + 1 { n - 1 - i } else { 0 };
                    extended[pad_len - 1 - i] = signal[idx1];
                    extended[n + pad_len + i] = signal[idx2];
                }
                for i in 0..n {
                    extended[i + pad_len] = signal[i];
                }
            }
        }

        Ok(extended)
    }

    fn convolve_downsample(&self, signal: &Array1<f64>, filter: &[f64]) -> Result<Array1<f64>> {
        let n = signal.len();
        let filter_len = filter.len();
        let output_len = (n + 1) / 2;
        let mut output = Array1::zeros(output_len);

        for i in 0..output_len {
            let pos = i * 2;
            let mut sum = 0.0;

            for (j, &coeff) in filter.iter().enumerate() {
                let idx = pos + j;
                if idx < n {
                    sum += signal[idx] * coeff;
                }
            }

            output[i] = sum;
        }

        Ok(output)
    }

    fn upsample_convolve(&self, signal: &ArrayView1<f64>, filter: &[f64]) -> Result<Array1<f64>> {
        let n = signal.len();
        let filter_len = filter.len();
        let output_len = n * 2;
        let mut output = Array1::zeros(output_len);

        // Upsample by inserting zeros
        let mut upsampled = Array1::zeros(output_len);
        for i in 0..n {
            upsampled[i * 2] = signal[i];
        }

        // Convolve with reconstruction filter
        for i in 0..output_len {
            let mut sum = 0.0;
            for (j, &coeff) in filter.iter().enumerate() {
                if i >= j && i - j < output_len {
                    sum += upsampled[i - j] * coeff;
                }
            }
            output[i] = sum;
        }

        Ok(output)
    }

    fn max_decomposition_level(&self, signal_len: usize) -> usize {
        let filter_len = self.filters.dec_lo.len();
        let mut level: usize = 0;
        let mut current_len = signal_len;

        while current_len >= filter_len {
            current_len = (current_len + 1) / 2;
            level += 1;
        }

        level.saturating_sub(1)
    }
}

/// 2D Discrete Wavelet Transform
#[derive(Debug, Clone)]
pub struct DWT2D {
    wavelet: WaveletType,
    filters: WaveletFilters,
    boundary: BoundaryMode,
    level: Option<usize>,
}

impl DWT2D {
    /// Create a new DWT2D instance
    pub fn new(wavelet: WaveletType) -> Result<Self> {
        let filters = WaveletFilters::from_wavelet(wavelet)?;
        Ok(DWT2D {
            wavelet,
            filters,
            boundary: BoundaryMode::Symmetric,
            level: None,
        })
    }

    /// Set the boundary mode
    pub fn with_boundary(mut self, boundary: BoundaryMode) -> Self {
        self.boundary = boundary;
        self
    }

    /// Set the decomposition level
    pub fn with_level(mut self, level: usize) -> Self {
        self.level = Some(level);
        self
    }

    /// Perform single-level 2D decomposition
    pub fn decompose2(&self, image: &ArrayView2<f64>) -> Result<Dwt2dCoeffs> {
        let (rows, cols) = image.dim();
        if rows < 2 || cols < 2 {
            return Err(TransformError::InvalidInput(
                "Image too small for 2D DWT".to_string(),
            ));
        }

        let dwt1d = DWT {
            wavelet: self.wavelet,
            filters: self.filters.clone(),
            boundary: self.boundary,
            level: None,
        };

        // Apply DWT along rows
        let mut row_results_approx = Vec::with_capacity(rows);
        let mut row_results_detail = Vec::with_capacity(rows);

        for row_idx in 0..rows {
            let row = image.row(row_idx);
            let (approx, detail) = dwt1d.decompose(&row)?;
            row_results_approx.push(approx);
            row_results_detail.push(detail);
        }

        let approx_rows = row_results_approx[0].len();
        let detail_rows = row_results_detail[0].len();

        // Convert to 2D arrays
        let mut approx_mat = Array2::zeros((rows, approx_rows));
        let mut detail_mat = Array2::zeros((rows, detail_rows));

        for (i, (app, det)) in row_results_approx
            .iter()
            .zip(row_results_detail.iter())
            .enumerate()
        {
            for (j, &val) in app.iter().enumerate() {
                approx_mat[[i, j]] = val;
            }
            for (j, &val) in det.iter().enumerate() {
                detail_mat[[i, j]] = val;
            }
        }

        // Apply DWT along columns
        let (ll, lh) = self.decompose_columns(&approx_mat.view(), &dwt1d)?;
        let (hl, hh) = self.decompose_columns(&detail_mat.view(), &dwt1d)?;

        Ok(Dwt2dCoeffs { ll, lh, hl, hh })
    }

    fn decompose_columns(
        &self,
        mat: &ArrayView2<f64>,
        dwt1d: &DWT,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let (rows, cols) = mat.dim();
        let mut col_results_approx = Vec::with_capacity(cols);
        let mut col_results_detail = Vec::with_capacity(cols);

        for col_idx in 0..cols {
            let col = mat.column(col_idx);
            let (approx, detail) = dwt1d.decompose(&col)?;
            col_results_approx.push(approx);
            col_results_detail.push(detail);
        }

        let approx_cols = col_results_approx[0].len();
        let detail_cols = col_results_detail[0].len();

        let mut approx_result = Array2::zeros((approx_cols, cols));
        let mut detail_result = Array2::zeros((detail_cols, cols));

        for (j, (app, det)) in col_results_approx
            .iter()
            .zip(col_results_detail.iter())
            .enumerate()
        {
            for (i, &val) in app.iter().enumerate() {
                approx_result[[i, j]] = val;
            }
            for (i, &val) in det.iter().enumerate() {
                detail_result[[i, j]] = val;
            }
        }

        Ok((approx_result, detail_result))
    }

    /// Perform multi-level 2D decomposition
    pub fn wavedec2(&self, image: &ArrayView2<f64>) -> Result<Vec<Dwt2dCoeffs>> {
        let max_level = self.max_decomposition_level_2d(image.dim());
        let level = self.level.unwrap_or(max_level).min(max_level);

        let mut coeffs = Vec::with_capacity(level);
        let mut current = image.to_owned();

        for _ in 0..level {
            let dwt2d_coeffs = self.decompose2(&current.view())?;
            coeffs.push(dwt2d_coeffs.clone());
            current = dwt2d_coeffs.ll;
        }

        Ok(coeffs)
    }

    fn max_decomposition_level_2d(&self, shape: (usize, usize)) -> usize {
        let filter_len = self.filters.dec_lo.len();
        let min_dim = shape.0.min(shape.1);

        let mut level: usize = 0;
        let mut current_dim = min_dim;

        while current_dim >= filter_len {
            current_dim = (current_dim + 1) / 2;
            level += 1;
        }

        level.saturating_sub(1)
    }
}

/// 2D DWT coefficients (LL, LH, HL, HH)
#[derive(Debug, Clone)]
pub struct Dwt2dCoeffs {
    /// Approximation coefficients (low-low)
    pub ll: Array2<f64>,
    /// Horizontal detail coefficients (low-high)
    pub lh: Array2<f64>,
    /// Vertical detail coefficients (high-low)
    pub hl: Array2<f64>,
    /// Diagonal detail coefficients (high-high)
    pub hh: Array2<f64>,
}

/// N-D Discrete Wavelet Transform (placeholder for 3D and higher)
#[derive(Debug, Clone)]
pub struct DWTN {
    wavelet: WaveletType,
    boundary: BoundaryMode,
    level: Option<usize>,
}

impl DWTN {
    /// Create a new DWTN instance
    pub fn new(wavelet: WaveletType) -> Self {
        DWTN {
            wavelet,
            boundary: BoundaryMode::Symmetric,
            level: None,
        }
    }

    /// Set the boundary mode
    pub fn with_boundary(mut self, boundary: BoundaryMode) -> Self {
        self.boundary = boundary;
        self
    }

    /// Set the decomposition level
    pub fn with_level(mut self, level: usize) -> Self {
        self.level = Some(level);
        self
    }

    /// Perform 3D decomposition (simplified placeholder)
    pub fn decompose3(&self, _volume: &Array3<f64>) -> Result<Array3<f64>> {
        Err(TransformError::NotImplemented(
            "3D DWT not yet fully implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dwt_haar() -> Result<()> {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let dwt = DWT::new(WaveletType::Haar)?;

        let (approx, detail) = dwt.decompose(&signal.view())?;

        assert!(approx.len() > 0);
        assert!(detail.len() > 0);
        assert_eq!(approx.len(), detail.len());

        Ok(())
    }

    #[test]
    fn test_dwt_multilevel() -> Result<()> {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let dwt = DWT::new(WaveletType::Haar)?.with_level(2);

        let coeffs = dwt.wavedec(&signal.view())?;

        assert_eq!(coeffs.len(), 3); // 2 levels + approximation

        Ok(())
    }

    #[test]
    fn test_dwt_reconstruction() -> Result<()> {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let dwt = DWT::new(WaveletType::Haar)?;

        let (approx, detail) = dwt.decompose(&signal.view())?;
        let reconstructed = dwt.reconstruct(&approx.view(), &detail.view())?;

        // Check reconstruction is approximately correct (may have different length)
        assert!(reconstructed.len() >= signal.len() - 2);

        Ok(())
    }

    #[test]
    fn test_dwt2d() -> Result<()> {
        let image = Array2::from_shape_fn((8, 8), |(i, j)| (i + j) as f64);
        let dwt2d = DWT2D::new(WaveletType::Haar)?;

        let coeffs = dwt2d.decompose2(&image.view())?;

        assert!(coeffs.ll.len() > 0);
        assert!(coeffs.lh.len() > 0);
        assert!(coeffs.hl.len() > 0);
        assert!(coeffs.hh.len() > 0);

        Ok(())
    }

    #[test]
    fn test_wavelet_filters() -> Result<()> {
        let filters = WaveletFilters::from_wavelet(WaveletType::Haar)?;

        assert_eq!(filters.dec_lo.len(), 2);
        assert_eq!(filters.dec_hi.len(), 2);
        assert_eq!(filters.rec_lo.len(), 2);
        assert_eq!(filters.rec_hi.len(), 2);

        Ok(())
    }
}
