//! FastText embeddings with character n-grams
//!
//! This module implements FastText, an extension of Word2Vec that learns
//! word representations as bags of character n-grams. This approach handles
//! out-of-vocabulary words and morphologically rich languages better.
//!
//! ## Overview
//!
//! FastText represents each word as a bag of character n-grams. For example:
//! - word: "where"
//! - 3-grams: "<wh", "whe", "her", "ere", "re>"
//! - The word embedding is the sum of its n-gram embeddings
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_text::embeddings::fasttext::{FastText, FastTextConfig};
//!
//! // Create configuration
//! let config = FastTextConfig {
//!     vector_size: 100,
//!     min_n: 3,
//!     max_n: 6,
//!     window_size: 5,
//!     epochs: 5,
//!     learning_rate: 0.05,
//!     min_count: 1,
//!     negative_samples: 5,
//!     ..Default::default()
//! };
//!
//! // Train model
//! let documents = vec![
//!     "the quick brown fox jumps over the lazy dog",
//!     "a quick brown dog outpaces a quick fox"
//! ];
//!
//! let mut model = FastText::with_config(config);
//! model.train(&documents).expect("Training failed");
//!
//! // Get word vector (works even for OOV words!)
//! if let Ok(vector) = model.get_word_vector("quickest") {
//!     println!("Vector for OOV word 'quickest': {:?}", vector);
//! }
//! ```

use crate::error::{Result, TextError};
use crate::tokenize::{Tokenizer, WordTokenizer};
use crate::vocabulary::Vocabulary;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// FastText configuration
#[derive(Debug, Clone)]
pub struct FastTextConfig {
    /// Size of word vectors
    pub vector_size: usize,
    /// Minimum length of character n-grams
    pub min_n: usize,
    /// Maximum length of character n-grams
    pub max_n: usize,
    /// Size of context window
    pub window_size: usize,
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Minimum word count threshold
    pub min_count: usize,
    /// Number of negative samples
    pub negative_samples: usize,
    /// Subsampling threshold for frequent words
    pub subsample: f64,
    /// Bucket size for hashing n-grams
    pub bucket_size: usize,
}

impl Default for FastTextConfig {
    fn default() -> Self {
        Self {
            vector_size: 100,
            min_n: 3,
            max_n: 6,
            window_size: 5,
            epochs: 5,
            learning_rate: 0.05,
            min_count: 5,
            negative_samples: 5,
            subsample: 1e-3,
            bucket_size: 2_000_000,
        }
    }
}

/// FastText model for learning word representations with character n-grams
pub struct FastText {
    /// Configuration
    config: FastTextConfig,
    /// Vocabulary of words
    vocabulary: Vocabulary,
    /// Word frequencies
    word_counts: HashMap<String, usize>,
    /// Word embeddings (for words in vocabulary)
    word_embeddings: Option<Array2<f64>>,
    /// N-gram embeddings (subword information)
    ngram_embeddings: Option<Array2<f64>>,
    /// N-gram to bucket index mapping
    ngram_to_bucket: HashMap<String, usize>,
    /// Tokenizer
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    /// Current learning rate
    current_learning_rate: f64,
}

impl Debug for FastText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastText")
            .field("config", &self.config)
            .field("vocabulary_size", &self.vocabulary.len())
            .field("word_embeddings", &self.word_embeddings.is_some())
            .field("ngram_embeddings", &self.ngram_embeddings.is_some())
            .finish()
    }
}

impl Clone for FastText {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            vocabulary: self.vocabulary.clone(),
            word_counts: self.word_counts.clone(),
            word_embeddings: self.word_embeddings.clone(),
            ngram_embeddings: self.ngram_embeddings.clone(),
            ngram_to_bucket: self.ngram_to_bucket.clone(),
            tokenizer: Box::new(WordTokenizer::default()),
            current_learning_rate: self.current_learning_rate,
        }
    }
}

impl FastText {
    /// Create a new FastText model with default configuration
    pub fn new() -> Self {
        Self {
            config: FastTextConfig::default(),
            vocabulary: Vocabulary::new(),
            word_counts: HashMap::new(),
            word_embeddings: None,
            ngram_embeddings: None,
            ngram_to_bucket: HashMap::new(),
            tokenizer: Box::new(WordTokenizer::default()),
            current_learning_rate: 0.05,
        }
    }

    /// Create a new FastText model with custom configuration
    pub fn with_config(config: FastTextConfig) -> Self {
        let learning_rate = config.learning_rate;
        Self {
            config,
            vocabulary: Vocabulary::new(),
            word_counts: HashMap::new(),
            word_embeddings: None,
            ngram_embeddings: None,
            ngram_to_bucket: HashMap::new(),
            tokenizer: Box::new(WordTokenizer::default()),
            current_learning_rate: learning_rate,
        }
    }

    /// Extract character n-grams from a word
    fn extract_ngrams(&self, word: &str) -> Vec<String> {
        let word_with_boundaries = format!("<{}>", word);
        let chars: Vec<char> = word_with_boundaries.chars().collect();
        let mut ngrams = Vec::new();

        for n in self.config.min_n..=self.config.max_n {
            if chars.len() < n {
                continue;
            }

            for i in 0..=(chars.len() - n) {
                let ngram: String = chars[i..i + n].iter().collect();
                ngrams.push(ngram);
            }
        }

        ngrams
    }

    /// Hash an n-gram to a bucket index
    fn hash_ngram(&self, ngram: &str) -> usize {
        // Simple hash function (FNV-1a)
        let mut hash: u64 = 2166136261;
        for byte in ngram.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(16777619);
        }
        (hash % (self.config.bucket_size as u64)) as usize
    }

    /// Build vocabulary from texts
    pub fn build_vocabulary(&mut self, texts: &[&str]) -> Result<()> {
        if texts.is_empty() {
            return Err(TextError::InvalidInput(
                "No texts provided for building vocabulary".into(),
            ));
        }

        // Count word frequencies
        let mut word_counts = HashMap::new();

        for &text in texts {
            let tokens = self.tokenizer.tokenize(text)?;
            for token in tokens {
                *word_counts.entry(token).or_insert(0) += 1;
            }
        }

        // Build vocabulary with min_count threshold
        self.vocabulary = Vocabulary::new();
        for (word, count) in &word_counts {
            if *count >= self.config.min_count {
                self.vocabulary.add_token(word);
            }
        }

        if self.vocabulary.is_empty() {
            return Err(TextError::VocabularyError(
                "No words meet the minimum count threshold".into(),
            ));
        }

        self.word_counts = word_counts;

        // Initialize embeddings
        let vocab_size = self.vocabulary.len();
        let vector_size = self.config.vector_size;
        let bucket_size = self.config.bucket_size;

        let mut rng = scirs2_core::random::rng();

        // Initialize word embeddings
        let word_embeddings = Array2::from_shape_fn((vocab_size, vector_size), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) / vector_size as f64
        });

        // Initialize n-gram embeddings
        let ngram_embeddings = Array2::from_shape_fn((bucket_size, vector_size), |_| {
            (rng.random::<f64>() * 2.0 - 1.0) / vector_size as f64
        });

        self.word_embeddings = Some(word_embeddings);
        self.ngram_embeddings = Some(ngram_embeddings);

        // Build n-gram to bucket mapping
        self.ngram_to_bucket.clear();
        for i in 0..self.vocabulary.len() {
            if let Some(word) = self.vocabulary.get_token(i) {
                let ngrams = self.extract_ngrams(word);
                for ngram in ngrams {
                    if !self.ngram_to_bucket.contains_key(&ngram) {
                        let bucket = self.hash_ngram(&ngram);
                        self.ngram_to_bucket.insert(ngram, bucket);
                    }
                }
            }
        }

        Ok(())
    }

    /// Train the FastText model
    pub fn train(&mut self, texts: &[&str]) -> Result<()> {
        if texts.is_empty() {
            return Err(TextError::InvalidInput(
                "No texts provided for training".into(),
            ));
        }

        // Build vocabulary if not already built
        if self.vocabulary.is_empty() {
            self.build_vocabulary(texts)?;
        }

        // Prepare training data
        let mut sentences = Vec::new();
        for &text in texts {
            let tokens = self.tokenizer.tokenize(text)?;
            let word_indices: Vec<usize> = tokens
                .iter()
                .filter_map(|token| self.vocabulary.get_index(token))
                .collect();
            if !word_indices.is_empty() {
                sentences.push(word_indices);
            }
        }

        // Training loop
        for epoch in 0..self.config.epochs {
            // Update learning rate
            self.current_learning_rate =
                self.config.learning_rate * (1.0 - (epoch as f64 / self.config.epochs as f64));
            self.current_learning_rate = self
                .current_learning_rate
                .max(self.config.learning_rate * 0.0001);

            // Train on each sentence
            for sentence in &sentences {
                self.train_sentence(sentence)?;
            }
        }

        Ok(())
    }

    /// Train on a single sentence
    fn train_sentence(&mut self, sentence: &[usize]) -> Result<()> {
        if sentence.len() < 2 {
            return Ok(());
        }

        // Extract all ngrams for all words in sentence BEFORE taking mutable borrows
        let mut sentence_ngrams = Vec::with_capacity(sentence.len());
        for &target_idx in sentence {
            let target_word = self
                .vocabulary
                .get_token(target_idx)
                .ok_or_else(|| TextError::VocabularyError("Invalid word index".into()))?;
            let ngrams = self.extract_ngrams(target_word);
            let ngram_buckets: Vec<usize> = ngrams
                .iter()
                .filter_map(|ng| self.ngram_to_bucket.get(ng).copied())
                .collect();
            sentence_ngrams.push(ngram_buckets);
        }

        let word_embeddings = self
            .word_embeddings
            .as_mut()
            .ok_or_else(|| TextError::EmbeddingError("Word embeddings not initialized".into()))?;
        let ngram_embeddings = self
            .ngram_embeddings
            .as_mut()
            .ok_or_else(|| TextError::EmbeddingError("N-gram embeddings not initialized".into()))?;

        let mut rng = scirs2_core::random::rng();

        // Skip-gram training for each word in the sentence
        for (pos, &target_idx) in sentence.iter().enumerate() {
            // Random window size
            let window = 1 + rng.random_range(0..self.config.window_size);

            // Get precomputed n-gram buckets for this target word
            let ngram_buckets = &sentence_ngrams[pos];

            // Average word embedding with n-gram embeddings
            let mut target_vec = word_embeddings.row(target_idx).to_owned();
            for &bucket in ngram_buckets {
                target_vec += &ngram_embeddings.row(bucket);
            }
            if !ngram_buckets.is_empty() {
                target_vec /= 1.0 + ngram_buckets.len() as f64;
            }

            // For each context word in window
            for i in pos.saturating_sub(window)..=(pos + window).min(sentence.len() - 1) {
                if i == pos {
                    continue;
                }

                let context_idx = sentence[i];

                // Positive example
                let context_vec = word_embeddings.row(context_idx).to_owned();
                let dot_product: f64 = target_vec
                    .iter()
                    .zip(context_vec.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
                let gradient = (1.0 - sigmoid) * self.current_learning_rate;

                // Update embeddings
                let update = &target_vec * gradient;
                let mut context_row = word_embeddings.row_mut(context_idx);
                context_row += &update;

                // Update n-gram embeddings (pre-compute scaled update to avoid move in loop)
                if !ngram_buckets.is_empty() {
                    let ngram_update = update / (1.0 + ngram_buckets.len() as f64);
                    for &bucket in ngram_buckets {
                        let mut ngram_row = ngram_embeddings.row_mut(bucket);
                        ngram_row += &ngram_update;
                    }
                }

                // Negative sampling
                for _ in 0..self.config.negative_samples {
                    let neg_idx = rng.random_range(0..self.vocabulary.len());
                    if neg_idx == context_idx {
                        continue;
                    }

                    let neg_vec = word_embeddings.row(neg_idx).to_owned();
                    let dot_product: f64 = target_vec
                        .iter()
                        .zip(neg_vec.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
                    let gradient = -sigmoid * self.current_learning_rate;

                    let update = &target_vec * gradient;
                    let mut neg_row = word_embeddings.row_mut(neg_idx);
                    neg_row += &update;
                }
            }
        }

        Ok(())
    }

    /// Get the embedding vector for a word (handles OOV words)
    pub fn get_word_vector(&self, word: &str) -> Result<Array1<f64>> {
        let word_embeddings = self
            .word_embeddings
            .as_ref()
            .ok_or_else(|| TextError::EmbeddingError("Model not trained".into()))?;
        let ngram_embeddings = self
            .ngram_embeddings
            .as_ref()
            .ok_or_else(|| TextError::EmbeddingError("Model not trained".into()))?;

        let ngrams = self.extract_ngrams(word);
        let mut vector = Array1::zeros(self.config.vector_size);
        let mut count = 0.0;

        // Add word embedding if in vocabulary
        if let Some(idx) = self.vocabulary.get_index(word) {
            vector += &word_embeddings.row(idx);
            count += 1.0;
        }

        // Add n-gram embeddings
        for ngram in &ngrams {
            if let Some(&bucket) = self.ngram_to_bucket.get(ngram) {
                vector += &ngram_embeddings.row(bucket);
                count += 1.0;
            }
        }

        if count > 0.0 {
            vector /= count;
            Ok(vector)
        } else {
            Err(TextError::VocabularyError(format!(
                "Cannot compute vector for word '{}': no n-grams found",
                word
            )))
        }
    }

    /// Find most similar words
    pub fn most_similar(&self, word: &str, top_n: usize) -> Result<Vec<(String, f64)>> {
        let word_vec = self.get_word_vector(word)?;
        let word_embeddings = self
            .word_embeddings
            .as_ref()
            .ok_or_else(|| TextError::EmbeddingError("Model not trained".into()))?;

        let mut similarities = Vec::new();

        for i in 0..self.vocabulary.len() {
            if let Some(candidate) = self.vocabulary.get_token(i) {
                if candidate == word {
                    continue;
                }

                let candidate_vec = word_embeddings.row(i).to_owned();
                let similarity = cosine_similarity(&word_vec, &candidate_vec);
                similarities.push((candidate.to_string(), similarity));
            }
        }

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(similarities.into_iter().take(top_n).collect())
    }

    /// Save the model to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let word_embeddings = self
            .word_embeddings
            .as_ref()
            .ok_or_else(|| TextError::EmbeddingError("Model not trained".into()))?;

        let mut file = File::create(path).map_err(|e| TextError::IoError(e.to_string()))?;

        // Write header
        writeln!(
            &mut file,
            "{} {}",
            self.vocabulary.len(),
            self.config.vector_size
        )
        .map_err(|e| TextError::IoError(e.to_string()))?;

        // Write each word and its vector
        for i in 0..self.vocabulary.len() {
            if let Some(word) = self.vocabulary.get_token(i) {
                write!(&mut file, "{} ", word).map_err(|e| TextError::IoError(e.to_string()))?;

                for j in 0..self.config.vector_size {
                    write!(&mut file, "{:.6} ", word_embeddings[[i, j]])
                        .map_err(|e| TextError::IoError(e.to_string()))?;
                }

                writeln!(&mut file).map_err(|e| TextError::IoError(e.to_string()))?;
            }
        }

        Ok(())
    }

    /// Get the vocabulary size
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Get the vector size
    pub fn vector_size(&self) -> usize {
        self.config.vector_size
    }
}

impl Default for FastText {
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

    #[test]
    fn test_extract_ngrams() {
        let config = FastTextConfig {
            min_n: 3,
            max_n: 4,
            ..Default::default()
        };
        let model = FastText::with_config(config);

        let ngrams = model.extract_ngrams("test");
        // Should include: "<te", "tes", "est", "st>", "<tes", "test", "est>", ...
        assert!(!ngrams.is_empty());
        assert!(ngrams.contains(&"<te".to_string()));
        assert!(ngrams.contains(&"est".to_string()));
    }

    #[test]
    fn test_fasttext_training() {
        let texts = [
            "the quick brown fox jumps over the lazy dog",
            "a quick brown dog outpaces a quick fox",
        ];

        let config = FastTextConfig {
            vector_size: 10,
            window_size: 2,
            min_count: 1,
            epochs: 1,
            min_n: 3,
            max_n: 4,
            ..Default::default()
        };

        let mut model = FastText::with_config(config);
        let result = model.train(&texts);
        assert!(result.is_ok());

        // Test getting vector for in-vocabulary word
        let vec = model.get_word_vector("quick");
        assert!(vec.is_ok());
        assert_eq!(vec.expect("Failed to get vector").len(), 10);

        // Test getting vector for OOV word (should work due to n-grams)
        let oov_vec = model.get_word_vector("quickest");
        assert!(oov_vec.is_ok());
    }

    #[test]
    fn test_fasttext_oov_handling() {
        let texts = ["hello world", "hello there"];

        let config = FastTextConfig {
            vector_size: 10,
            min_count: 1,
            epochs: 1,
            ..Default::default()
        };

        let mut model = FastText::with_config(config);
        model.train(&texts).expect("Training failed");

        // Get vector for OOV word that shares n-grams with "hello"
        let oov_vec = model.get_word_vector("helloworld");
        assert!(oov_vec.is_ok(), "FastText should handle OOV words");
    }
}
