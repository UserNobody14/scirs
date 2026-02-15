//! GloVe (Global Vectors for Word Representation) embeddings loader
//!
//! This module provides functionality to load pre-trained GloVe embeddings
//! and use them for downstream NLP tasks.
//!
//! ## Overview
//!
//! GloVe is an unsupervised learning algorithm for obtaining vector representations
//! for words. Training is performed on aggregated global word-word co-occurrence
//! statistics from a corpus.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use scirs2_text::embeddings::glove::GloVe;
//!
//! // Load pre-trained GloVe embeddings
//! let glove = GloVe::load("path/to/glove.6B.100d.txt")
//!     .expect("Failed to load GloVe embeddings");
//!
//! // Get word vector
//! if let Ok(vector) = glove.get_word_vector("king") {
//!     println!("Vector for 'king': {:?}", vector);
//! }
//!
//! // Find similar words
//! let similar = glove.most_similar("king", 5).expect("Failed to find similar words");
//! for (word, similarity) in similar {
//!     println!("{}: {:.4}", word, similarity);
//! }
//!
//! // Analogy: king - man + woman = ?
//! let analogy = glove.analogy("king", "man", "woman", 5)
//!     .expect("Failed to compute analogy");
//! println!("Analogy results: {:?}", analogy);
//! ```

use crate::error::{Result, TextError};
use crate::vocabulary::Vocabulary;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// GloVe embeddings model
///
/// Provides functionality to load and use pre-trained GloVe word embeddings.
#[derive(Debug, Clone)]
pub struct GloVe {
    /// Word vocabulary
    vocabulary: Vocabulary,
    /// Word to index mapping for fast lookup
    word_to_idx: HashMap<String, usize>,
    /// Embedding matrix where each row is a word vector
    embeddings: Array2<f64>,
    /// Dimensionality of the embeddings
    vector_size: usize,
}

impl GloVe {
    /// Create a new empty GloVe model
    pub fn new() -> Self {
        Self {
            vocabulary: Vocabulary::new(),
            word_to_idx: HashMap::new(),
            embeddings: Array2::zeros((0, 0)),
            vector_size: 0,
        }
    }

    /// Load GloVe embeddings from a file
    ///
    /// The file should be in the standard GloVe format:
    /// ```text
    /// word1 0.123 0.456 0.789 ...
    /// word2 0.234 0.567 0.890 ...
    /// ```
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GloVe file
    ///
    /// # Returns
    ///
    /// A new GloVe model with loaded embeddings
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(|e| TextError::IoError(e.to_string()))?;
        let reader = BufReader::new(file);

        let mut words = Vec::new();
        let mut vectors = Vec::new();
        let mut vector_size = 0;

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|e| TextError::IoError(e.to_string()))?;
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.is_empty() {
                continue;
            }

            // First part is the word, rest are vector components
            let word = parts[0].to_string();
            let vector_parts = &parts[1..];

            // Set vector size from first line
            if line_num == 0 {
                vector_size = vector_parts.len();
                if vector_size == 0 {
                    return Err(TextError::EmbeddingError(
                        "Invalid GloVe file: no vector components found".into(),
                    ));
                }
            }

            // Verify vector size consistency
            if vector_parts.len() != vector_size {
                return Err(TextError::EmbeddingError(format!(
                    "Inconsistent vector size at line {}: expected {}, got {}",
                    line_num + 1,
                    vector_size,
                    vector_parts.len()
                )));
            }

            // Parse vector components
            let vector: Result<Vec<f64>> = vector_parts
                .iter()
                .map(|&s| {
                    s.parse::<f64>().map_err(|_| {
                        TextError::EmbeddingError(format!(
                            "Failed to parse float at line {}: '{}'",
                            line_num + 1,
                            s
                        ))
                    })
                })
                .collect();

            words.push(word);
            vectors.push(vector?);
        }

        if words.is_empty() {
            return Err(TextError::EmbeddingError(
                "No embeddings loaded from file".into(),
            ));
        }

        // Build vocabulary
        let mut vocabulary = Vocabulary::new();
        let mut word_to_idx = HashMap::new();

        for (idx, word) in words.iter().enumerate() {
            vocabulary.add_token(word);
            word_to_idx.insert(word.clone(), idx);
        }

        // Create embedding matrix
        let vocab_size = words.len();
        let mut embeddings = Array2::zeros((vocab_size, vector_size));

        for (idx, vector) in vectors.iter().enumerate() {
            for (j, &val) in vector.iter().enumerate() {
                embeddings[[idx, j]] = val;
            }
        }

        Ok(Self {
            vocabulary,
            word_to_idx,
            embeddings,
            vector_size,
        })
    }

    /// Save GloVe embeddings to a file
    ///
    /// Saves in the standard GloVe format compatible with load()
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file = File::create(path).map_err(|e| TextError::IoError(e.to_string()))?;

        for (word, &idx) in &self.word_to_idx {
            write!(&mut file, "{}", word).map_err(|e| TextError::IoError(e.to_string()))?;

            for j in 0..self.vector_size {
                write!(&mut file, " {:.6}", self.embeddings[[idx, j]])
                    .map_err(|e| TextError::IoError(e.to_string()))?;
            }

            writeln!(&mut file).map_err(|e| TextError::IoError(e.to_string()))?;
        }

        Ok(())
    }

    /// Get the vector for a word
    ///
    /// # Arguments
    ///
    /// * `word` - The word to look up
    ///
    /// # Returns
    ///
    /// The embedding vector for the word, or an error if not found
    pub fn get_word_vector(&self, word: &str) -> Result<Array1<f64>> {
        match self.word_to_idx.get(word) {
            Some(&idx) => Ok(self.embeddings.row(idx).to_owned()),
            None => Err(TextError::VocabularyError(format!(
                "Word '{}' not in vocabulary",
                word
            ))),
        }
    }

    /// Find the most similar words to a given word
    ///
    /// # Arguments
    ///
    /// * `word` - The query word
    /// * `top_n` - Number of similar words to return
    ///
    /// # Returns
    ///
    /// A vector of (word, similarity) pairs, sorted by similarity (descending)
    pub fn most_similar(&self, word: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        let word_vec = self.get_word_vector(word)?;
        self.most_similar_by_vector(&word_vec, top_n, &[word])
    }

    /// Find the most similar words to a given vector
    ///
    /// # Arguments
    ///
    /// * `vector` - The query vector
    /// * `top_n` - Number of similar words to return
    /// * `exclude_words` - Words to exclude from results
    ///
    /// # Returns
    ///
    /// A vector of (word, similarity) pairs, sorted by similarity (descending)
    pub fn most_similar_by_vector(
        &self,
        vector: &Array1<f64>,
        top_n: usize,
        exclude_words: &[&str],
    ) -> Result<Vec<(String, f64)>> {
        // Create set of excluded indices
        let exclude_indices: Vec<usize> = exclude_words
            .iter()
            .filter_map(|&word| self.word_to_idx.get(word).copied())
            .collect();

        // Calculate cosine similarity for all words
        let mut similarities = Vec::new();

        for (word, &idx) in &self.word_to_idx {
            if exclude_indices.contains(&idx) {
                continue;
            }

            let word_vec = self.embeddings.row(idx).to_owned();
            let similarity = cosine_similarity(vector, &word_vec);
            similarities.push((word.clone(), similarity));
        }

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top N
        Ok(similarities.into_iter().take(top_n).collect())
    }

    /// Compute word analogy: a is to b as c is to ?
    ///
    /// # Arguments
    ///
    /// * `a` - First word in the analogy
    /// * `b` - Second word in the analogy
    /// * `c` - Third word in the analogy
    /// * `top_n` - Number of results to return
    ///
    /// # Returns
    ///
    /// A vector of (word, similarity) pairs representing possible answers
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use scirs2_text::embeddings::glove::GloVe;
    /// # let glove = GloVe::new();
    /// // king - man + woman = ?
    /// let result = glove.analogy("king", "man", "woman", 5);
    /// // Expected result: "queen" should be among top results
    /// ```
    pub fn analogy(&self, a: &str, b: &str, c: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        // Get vectors
        let a_vec = self.get_word_vector(a)?;
        let b_vec = self.get_word_vector(b)?;
        let c_vec = self.get_word_vector(c)?;

        // Compute: b - a + c
        let mut result_vec = b_vec.clone();
        result_vec -= &a_vec;
        result_vec += &c_vec;

        // Normalize the result vector
        let norm = result_vec.iter().fold(0.0, |sum, &x| sum + x * x).sqrt();
        if norm > 0.0 {
            result_vec.mapv_inplace(|x| x / norm);
        }

        // Find most similar words to the result vector
        self.most_similar_by_vector(&result_vec, top_n, &[a, b, c])
    }

    /// Get the vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.word_to_idx.len()
    }

    /// Get the vector dimensionality
    pub fn vector_size(&self) -> usize {
        self.vector_size
    }

    /// Check if a word is in the vocabulary
    pub fn contains(&self, word: &str) -> bool {
        self.word_to_idx.contains_key(word)
    }

    /// Get all words in the vocabulary
    pub fn get_words(&self) -> Vec<String> {
        self.word_to_idx.keys().cloned().collect()
    }

    /// Get the embeddings matrix (for advanced use cases)
    pub fn get_embeddings(&self) -> &Array2<f64> {
        &self.embeddings
    }
}

impl Default for GloVe {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_glove_load_save() {
        // Create a temporary GloVe file
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(temp_file, "king 0.1 0.2 0.3").expect("Failed to write");
        writeln!(temp_file, "queen 0.15 0.25 0.35").expect("Failed to write");
        writeln!(temp_file, "man 0.05 0.1 0.15").expect("Failed to write");
        writeln!(temp_file, "woman 0.08 0.13 0.18").expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        // Load the GloVe model
        let glove = GloVe::load(temp_file.path()).expect("Failed to load GloVe");

        // Check vocabulary size and vector size
        assert_eq!(glove.vocabulary_size(), 4);
        assert_eq!(glove.vector_size(), 3);

        // Check word lookup
        assert!(glove.contains("king"));
        assert!(glove.contains("queen"));
        assert!(!glove.contains("prince"));

        // Get word vector
        let king_vec = glove.get_word_vector("king").expect("Failed to get vector");
        assert_eq!(king_vec.len(), 3);
        assert!((king_vec[0] - 0.1).abs() < 1e-6);
        assert!((king_vec[1] - 0.2).abs() < 1e-6);
        assert!((king_vec[2] - 0.3).abs() < 1e-6);

        // Test save and reload
        let save_path = std::env::temp_dir().join("test_glove_save.txt");
        glove.save(&save_path).expect("Failed to save");

        let reloaded = GloVe::load(&save_path).expect("Failed to reload");
        assert_eq!(reloaded.vocabulary_size(), glove.vocabulary_size());
        assert_eq!(reloaded.vector_size(), glove.vector_size());

        // Cleanup
        std::fs::remove_file(save_path).ok();
    }

    #[test]
    fn test_glove_similarity() {
        // Create a temporary GloVe file with similar vectors
        let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
        writeln!(temp_file, "king 1.0 0.0 0.0").expect("Failed to write");
        writeln!(temp_file, "queen 0.9 0.1 0.0").expect("Failed to write");
        writeln!(temp_file, "man 0.5 0.5 0.0").expect("Failed to write");
        writeln!(temp_file, "woman 0.4 0.6 0.0").expect("Failed to write");
        writeln!(temp_file, "cat 0.0 0.0 1.0").expect("Failed to write");
        temp_file.flush().expect("Failed to flush");

        let glove = GloVe::load(temp_file.path()).expect("Failed to load GloVe");

        // Find similar words to "king"
        let similar = glove
            .most_similar("king", 2)
            .expect("Failed to find similar");

        // "queen" should be most similar to "king"
        assert_eq!(similar.len(), 2);
        assert_eq!(similar[0].0, "queen");
        assert!(similar[0].1 > 0.9);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);

        let a = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let b = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![-1.0, 0.0, 0.0]);
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }
}
