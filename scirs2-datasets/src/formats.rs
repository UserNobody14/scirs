//! Support for various data formats (Parquet, Arrow, HDF5)
//!
//! This module provides integration with scirs2-io for reading and writing
//! datasets in modern columnar formats like Parquet and Arrow, as well as
//! scientific formats like HDF5, with memory-efficient streaming support.

#[cfg(feature = "formats")]
use crate::error::{DatasetsError, Result};
#[cfg(feature = "formats")]
use crate::utils::Dataset;
#[cfg(feature = "formats")]
use scirs2_core::ndarray::{Array1, Array2};
#[cfg(feature = "formats")]
use std::path::Path;

/// Format type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatType {
    /// Apache Parquet columnar format
    Parquet,
    /// Apache Arrow in-memory format
    Arrow,
    /// HDF5 hierarchical format
    Hdf5,
    /// CSV format (for completeness)
    Csv,
}

impl FormatType {
    /// Detect format from file extension
    pub fn from_extension(path: &str) -> Option<Self> {
        let lower = path.to_lowercase();
        if lower.ends_with(".parquet") || lower.ends_with(".pq") {
            Some(FormatType::Parquet)
        } else if lower.ends_with(".arrow") {
            Some(FormatType::Arrow)
        } else if lower.ends_with(".h5") || lower.ends_with(".hdf5") {
            Some(FormatType::Hdf5)
        } else if lower.ends_with(".csv") {
            Some(FormatType::Csv)
        } else {
            None
        }
    }

    /// Get file extension for this format
    pub fn extension(&self) -> &'static str {
        match self {
            FormatType::Parquet => "parquet",
            FormatType::Arrow => "arrow",
            FormatType::Hdf5 => "h5",
            FormatType::Csv => "csv",
        }
    }
}

/// Configuration for format conversion
#[derive(Debug, Clone)]
pub struct FormatConfig {
    /// Chunk size for streaming operations
    pub chunk_size: usize,
    /// Compression codec
    pub compression: Option<CompressionCodec>,
    /// Whether to use memory mapping when possible
    pub use_mmap: bool,
    /// Buffer size for I/O operations
    pub buffer_size: usize,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            chunk_size: 10_000,
            compression: Some(CompressionCodec::Snappy),
            use_mmap: true,
            buffer_size: 8 * 1024 * 1024, // 8 MB
        }
    }
}

/// Compression codec options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionCodec {
    /// No compression
    None,
    /// Snappy compression
    Snappy,
    /// GZIP compression
    Gzip,
    /// LZ4 compression
    Lz4,
    /// ZSTD compression
    Zstd,
}

impl CompressionCodec {
    /// Get compression level (0-9 where applicable)
    pub fn level(&self) -> Option<i32> {
        match self {
            CompressionCodec::None | CompressionCodec::Snappy | CompressionCodec::Lz4 => None,
            CompressionCodec::Gzip => Some(6), // Default GZIP level
            CompressionCodec::Zstd => Some(3), // Default ZSTD level
        }
    }
}

// ============================================================================
// Parquet Support (when formats feature is enabled)
// ============================================================================

#[cfg(feature = "formats")]
/// Parquet reader for datasets
pub struct ParquetReader {
    config: FormatConfig,
}

#[cfg(feature = "formats")]
impl ParquetReader {
    /// Create a new Parquet reader
    pub fn new() -> Self {
        Self {
            config: FormatConfig::default(),
        }
    }

    /// Create a Parquet reader with custom configuration
    pub fn with_config(config: FormatConfig) -> Self {
        Self { config }
    }

    /// Read a Parquet file into a Dataset
    ///
    /// Note: This is a stub implementation. Full integration with scirs2-io
    /// would require coordinating with the scirs2-io Parquet implementation.
    pub fn read<P: AsRef<Path>>(&self, _path: P) -> Result<Dataset> {
        // TODO: Implement actual Parquet reading via scirs2-io
        // For now, return an error indicating feature is in development
        Err(DatasetsError::InvalidFormat(
            "Parquet reading requires scirs2-io parquet feature (in development)".to_string(),
        ))
    }
}

#[cfg(feature = "formats")]
impl Default for ParquetReader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "formats")]
/// Parquet writer for datasets
pub struct ParquetWriter {
    config: FormatConfig,
}

#[cfg(feature = "formats")]
impl ParquetWriter {
    /// Create a new Parquet writer
    pub fn new() -> Self {
        Self {
            config: FormatConfig::default(),
        }
    }

    /// Create a Parquet writer with custom configuration
    pub fn with_config(config: FormatConfig) -> Self {
        Self { config }
    }

    /// Write a Dataset to a Parquet file
    pub fn write<P: AsRef<Path>>(&self, _dataset: &Dataset, _path: P) -> Result<()> {
        // TODO: Implement actual Parquet writing via scirs2-io
        Err(DatasetsError::InvalidFormat(
            "Parquet writing requires scirs2-io parquet feature (in development)".to_string(),
        ))
    }
}

#[cfg(feature = "formats")]
impl Default for ParquetWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HDF5 Support
// ============================================================================

#[cfg(feature = "formats")]
/// HDF5 reader for datasets
pub struct Hdf5Reader {
    config: FormatConfig,
}

#[cfg(feature = "formats")]
impl Hdf5Reader {
    /// Create a new HDF5 reader
    pub fn new() -> Self {
        Self {
            config: FormatConfig::default(),
        }
    }

    /// Create an HDF5 reader with custom configuration
    pub fn with_config(config: FormatConfig) -> Self {
        Self { config }
    }

    /// Read an HDF5 file into a Dataset
    pub fn read<P: AsRef<Path>>(&self, _path: P, _dataset_name: &str) -> Result<Dataset> {
        // TODO: Implement HDF5 reading via scirs2-io
        Err(DatasetsError::InvalidFormat(
            "HDF5 reading requires scirs2-io hdf5 feature (in development)".to_string(),
        ))
    }
}

#[cfg(feature = "formats")]
impl Default for Hdf5Reader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "formats")]
/// HDF5 writer for datasets
pub struct Hdf5Writer {
    config: FormatConfig,
}

#[cfg(feature = "formats")]
impl Hdf5Writer {
    /// Create a new HDF5 writer
    pub fn new() -> Self {
        Self {
            config: FormatConfig::default(),
        }
    }

    /// Create an HDF5 writer with custom configuration
    pub fn with_config(config: FormatConfig) -> Self {
        Self { config }
    }

    /// Write a Dataset to an HDF5 file
    pub fn write<P: AsRef<Path>>(
        &self,
        _dataset: &Dataset,
        _path: P,
        _dataset_name: &str,
    ) -> Result<()> {
        // TODO: Implement HDF5 writing via scirs2-io
        Err(DatasetsError::InvalidFormat(
            "HDF5 writing requires scirs2-io hdf5 feature (in development)".to_string(),
        ))
    }
}

#[cfg(feature = "formats")]
impl Default for Hdf5Writer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Format Conversion
// ============================================================================

#[cfg(feature = "formats")]
/// Convert between different data formats
pub struct FormatConverter {
    config: FormatConfig,
}

#[cfg(feature = "formats")]
impl FormatConverter {
    /// Create a new format converter
    pub fn new() -> Self {
        Self {
            config: FormatConfig::default(),
        }
    }

    /// Convert a dataset from one format to another
    pub fn convert<P1: AsRef<Path>, P2: AsRef<Path>>(
        &self,
        input_path: P1,
        input_format: FormatType,
        output_path: P2,
        output_format: FormatType,
    ) -> Result<()> {
        // Read in input format
        let dataset = match input_format {
            FormatType::Parquet => ParquetReader::new().read(input_path)?,
            FormatType::Hdf5 => Hdf5Reader::new().read(input_path, "data")?,
            FormatType::Csv => {
                return Err(DatasetsError::InvalidFormat(
                    "CSV reading via format converter not yet implemented".to_string(),
                ))
            }
            FormatType::Arrow => {
                return Err(DatasetsError::InvalidFormat(
                    "Arrow format not yet supported".to_string(),
                ))
            }
        };

        // Write in output format
        match output_format {
            FormatType::Parquet => ParquetWriter::new().write(&dataset, output_path)?,
            FormatType::Hdf5 => Hdf5Writer::new().write(&dataset, output_path, "data")?,
            FormatType::Csv => {
                return Err(DatasetsError::InvalidFormat(
                    "CSV writing via format converter not yet implemented".to_string(),
                ))
            }
            FormatType::Arrow => {
                return Err(DatasetsError::InvalidFormat(
                    "Arrow format not yet supported".to_string(),
                ))
            }
        }

        Ok(())
    }

    /// Auto-detect format and read
    pub fn read_auto<P: AsRef<Path>>(&self, path: P) -> Result<Dataset> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| DatasetsError::InvalidFormat("Invalid path".to_string()))?;

        let format = FormatType::from_extension(path_str)
            .ok_or_else(|| DatasetsError::InvalidFormat("Could not detect format".to_string()))?;

        match format {
            FormatType::Parquet => ParquetReader::new().read(path),
            FormatType::Hdf5 => Hdf5Reader::new().read(path, "data"),
            _ => Err(DatasetsError::InvalidFormat(format!(
                "Unsupported format: {:?}",
                format
            ))),
        }
    }
}

#[cfg(feature = "formats")]
impl Default for FormatConverter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Read a Parquet file
#[cfg(feature = "formats")]
pub fn read_parquet<P: AsRef<Path>>(path: P) -> Result<Dataset> {
    ParquetReader::new().read(path)
}

/// Write a Parquet file
#[cfg(feature = "formats")]
pub fn write_parquet<P: AsRef<Path>>(dataset: &Dataset, path: P) -> Result<()> {
    ParquetWriter::new().write(dataset, path)
}

/// Read an HDF5 file
#[cfg(feature = "formats")]
pub fn read_hdf5<P: AsRef<Path>>(path: P, dataset_name: &str) -> Result<Dataset> {
    Hdf5Reader::new().read(path, dataset_name)
}

/// Write an HDF5 file
#[cfg(feature = "formats")]
pub fn write_hdf5<P: AsRef<Path>>(dataset: &Dataset, path: P, dataset_name: &str) -> Result<()> {
    Hdf5Writer::new().write(dataset, path, dataset_name)
}

/// Auto-detect format and read
#[cfg(feature = "formats")]
pub fn read_auto<P: AsRef<Path>>(path: P) -> Result<Dataset> {
    FormatConverter::new().read_auto(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(
            FormatType::from_extension("data.parquet"),
            Some(FormatType::Parquet)
        );
        assert_eq!(
            FormatType::from_extension("data.h5"),
            Some(FormatType::Hdf5)
        );
        assert_eq!(
            FormatType::from_extension("data.csv"),
            Some(FormatType::Csv)
        );
        assert_eq!(FormatType::from_extension("data.txt"), None);
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(FormatType::Parquet.extension(), "parquet");
        assert_eq!(FormatType::Hdf5.extension(), "h5");
        assert_eq!(FormatType::Csv.extension(), "csv");
    }

    #[test]
    fn test_compression_codec() {
        assert_eq!(CompressionCodec::None.level(), None);
        assert_eq!(CompressionCodec::Snappy.level(), None);
        assert_eq!(CompressionCodec::Gzip.level(), Some(6));
        assert_eq!(CompressionCodec::Zstd.level(), Some(3));
    }

    #[test]
    fn test_format_config() {
        let config = FormatConfig::default();
        assert_eq!(config.chunk_size, 10_000);
        assert_eq!(config.compression, Some(CompressionCodec::Snappy));
        assert!(config.use_mmap);
    }
}
