//! Wavelet Packet Transform (WPT) Implementation
//!
//! Provides wavelet packet decomposition and best basis selection.
//! Wavelet packets extend DWT by decomposing both approximation and detail coefficients.

use crate::error::{Result, TransformError};
use crate::signal_transforms::dwt::{BoundaryMode, WaveletType, DWT};
use scirs2_core::ndarray::{Array1, ArrayView1};
use std::collections::HashMap;

/// Wavelet packet node
#[derive(Debug, Clone)]
pub struct WaveletPacketNode {
    /// Node data (coefficients)
    pub data: Array1<f64>,
    /// Node path (sequence of 'a' for approximation, 'd' for detail)
    pub path: String,
    /// Level in the packet tree
    pub level: usize,
    /// Node index at this level
    pub index: usize,
    /// Cost/entropy of this node
    pub cost: f64,
}

impl WaveletPacketNode {
    /// Create a new wavelet packet node
    pub fn new(data: Array1<f64>, path: String, level: usize, index: usize) -> Self {
        let cost = Self::compute_cost(&data);
        WaveletPacketNode {
            data,
            path,
            level,
            index,
            cost,
        }
    }

    /// Compute the cost (Shannon entropy) of the node
    fn compute_cost(data: &Array1<f64>) -> f64 {
        let energy: f64 = data.iter().map(|x| x * x).sum();
        if energy < 1e-10 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &val in data.iter() {
            let p = (val * val) / energy;
            if p > 1e-10 {
                entropy -= p * p.ln();
            }
        }

        entropy
    }

    /// Update the cost
    pub fn update_cost(&mut self) {
        self.cost = Self::compute_cost(&self.data);
    }
}

/// Best basis selection criterion
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BestBasisCriterion {
    /// Shannon entropy
    Shannon,
    /// Threshold (number of coefficients above threshold)
    Threshold(f64),
    /// Log energy
    LogEnergy,
    /// Sure (Stein's Unbiased Risk Estimate)
    Sure,
}

/// Wavelet Packet Transform
#[derive(Debug, Clone)]
pub struct WPT {
    wavelet: WaveletType,
    max_level: usize,
    boundary: BoundaryMode,
    criterion: BestBasisCriterion,
    nodes: HashMap<String, WaveletPacketNode>,
}

impl WPT {
    /// Create a new WPT instance
    pub fn new(wavelet: WaveletType, max_level: usize) -> Self {
        WPT {
            wavelet,
            max_level,
            boundary: BoundaryMode::Symmetric,
            criterion: BestBasisCriterion::Shannon,
            nodes: HashMap::new(),
        }
    }

    /// Set the boundary mode
    pub fn with_boundary(mut self, boundary: BoundaryMode) -> Self {
        self.boundary = boundary;
        self
    }

    /// Set the best basis criterion
    pub fn with_criterion(mut self, criterion: BestBasisCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Perform full wavelet packet decomposition
    pub fn decompose(&mut self, signal: &ArrayView1<f64>) -> Result<()> {
        self.nodes.clear();

        // Create root node
        let root = WaveletPacketNode::new(signal.to_owned(), String::new(), 0, 0);
        self.nodes.insert(String::new(), root);

        // Recursively decompose
        self.decompose_node("", 0)?;

        Ok(())
    }

    /// Recursively decompose a node
    fn decompose_node(&mut self, path: &str, level: usize) -> Result<()> {
        if level >= self.max_level {
            return Ok(());
        }

        // Get the current node
        let node = self
            .nodes
            .get(path)
            .ok_or_else(|| TransformError::InvalidInput(format!("Node not found: {}", path)))?
            .clone();

        // Create DWT instance
        let dwt = DWT::new(self.wavelet)?.with_boundary(self.boundary);

        // Decompose
        let (approx, detail) = dwt.decompose(&node.data.view())?;

        // Create child nodes
        let approx_path = format!("{}a", path);
        let detail_path = format!("{}d", path);

        let index = node.index;
        let approx_node = WaveletPacketNode::new(approx, approx_path.clone(), level + 1, index * 2);
        let detail_node =
            WaveletPacketNode::new(detail, detail_path.clone(), level + 1, index * 2 + 1);

        self.nodes.insert(approx_path.clone(), approx_node);
        self.nodes.insert(detail_path.clone(), detail_node);

        // Recursively decompose child nodes
        self.decompose_node(&approx_path, level + 1)?;
        self.decompose_node(&detail_path, level + 1)?;

        Ok(())
    }

    /// Select the best basis using the specified criterion
    pub fn best_basis(&self) -> Result<Vec<WaveletPacketNode>> {
        let mut best_nodes = Vec::new();
        self.select_best_basis("", &mut best_nodes)?;
        Ok(best_nodes)
    }

    /// Recursively select best basis
    fn select_best_basis(&self, path: &str, selected: &mut Vec<WaveletPacketNode>) -> Result<f64> {
        let node = self
            .nodes
            .get(path)
            .ok_or_else(|| TransformError::InvalidInput(format!("Node not found: {}", path)))?;

        let approx_path = format!("{}a", path);
        let detail_path = format!("{}d", path);

        // Check if we have children
        if self.nodes.contains_key(&approx_path) && self.nodes.contains_key(&detail_path) {
            // Compute cost of decomposition
            let approx_cost = self.select_best_basis(&approx_path, selected)?;
            let detail_cost = self.select_best_basis(&detail_path, selected)?;
            let children_cost = approx_cost + detail_cost;

            // Compare with keeping this node
            if node.cost <= children_cost {
                // Keep this node
                selected.retain(|n| !n.path.starts_with(path) || n.path == path);
                selected.push(node.clone());
                Ok(node.cost)
            } else {
                // Use children
                Ok(children_cost)
            }
        } else {
            // Leaf node
            selected.push(node.clone());
            Ok(node.cost)
        }
    }

    /// Reconstruct signal from wavelet packet coefficients
    pub fn reconstruct(&self, nodes: &[WaveletPacketNode]) -> Result<Array1<f64>> {
        if nodes.is_empty() {
            return Err(TransformError::InvalidInput(
                "No nodes provided for reconstruction".to_string(),
            ));
        }

        // Find the root or reconstruct from best basis
        if let Some(root) = nodes.iter().find(|n| n.path.is_empty()) {
            return Ok(root.data.clone());
        }

        // For now, return error - full reconstruction requires inverse WPT
        Err(TransformError::NotImplemented(
            "Reconstruction from arbitrary basis not yet implemented".to_string(),
        ))
    }

    /// Get all nodes at a specific level
    pub fn get_level(&self, level: usize) -> Vec<&WaveletPacketNode> {
        self.nodes
            .values()
            .filter(|node| node.level == level)
            .collect()
    }

    /// Get a specific node by path
    pub fn get_node(&self, path: &str) -> Option<&WaveletPacketNode> {
        self.nodes.get(path)
    }

    /// Get all nodes
    pub fn nodes(&self) -> &HashMap<String, WaveletPacketNode> {
        &self.nodes
    }

    /// Compute the total cost of the best basis
    pub fn best_basis_cost(&self) -> Result<f64> {
        let best = self.best_basis()?;
        Ok(best.iter().map(|node| node.cost).sum())
    }
}

/// Denoise using wavelet packet transform
pub fn denoise_wpt(
    signal: &ArrayView1<f64>,
    wavelet: WaveletType,
    level: usize,
    threshold: f64,
) -> Result<Array1<f64>> {
    // Perform WPT
    let mut wpt = WPT::new(wavelet, level);
    wpt.decompose(signal)?;

    // Get best basis
    let best = wpt.best_basis()?;

    // Apply thresholding
    let mut denoised_nodes = Vec::new();
    for mut node in best {
        // Soft thresholding
        for val in node.data.iter_mut() {
            if val.abs() < threshold {
                *val = 0.0;
            } else {
                *val = if *val > 0.0 {
                    *val - threshold
                } else {
                    *val + threshold
                };
            }
        }
        node.update_cost();
        denoised_nodes.push(node);
    }

    // Reconstruct
    wpt.reconstruct(&denoised_nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_wpt_decompose() -> Result<()> {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut wpt = WPT::new(WaveletType::Haar, 2);

        wpt.decompose(&signal.view())?;

        // Should have nodes at levels 0, 1, 2
        assert!(wpt.get_node("").is_some());
        assert!(wpt.get_node("a").is_some());
        assert!(wpt.get_node("d").is_some());
        assert!(wpt.get_node("aa").is_some());
        assert!(wpt.get_node("ad").is_some());
        assert!(wpt.get_node("da").is_some());
        assert!(wpt.get_node("dd").is_some());

        Ok(())
    }

    #[test]
    fn test_wpt_best_basis() -> Result<()> {
        let signal = Array1::from_vec((0..16).map(|i| (i as f64 * 0.5).sin()).collect());
        let mut wpt = WPT::new(WaveletType::Haar, 3);

        wpt.decompose(&signal.view())?;
        let best = wpt.best_basis()?;

        assert!(!best.is_empty());

        // Check that all selected nodes are unique
        let mut paths: Vec<_> = best.iter().map(|n| n.path.clone()).collect();
        paths.sort();
        paths.dedup();
        assert_eq!(paths.len(), best.len());

        Ok(())
    }

    #[test]
    fn test_wpt_levels() -> Result<()> {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut wpt = WPT::new(WaveletType::Haar, 2);

        wpt.decompose(&signal.view())?;

        let level0 = wpt.get_level(0);
        let level1 = wpt.get_level(1);
        let level2 = wpt.get_level(2);

        assert_eq!(level0.len(), 1);
        assert_eq!(level1.len(), 2);
        assert_eq!(level2.len(), 4);

        Ok(())
    }

    #[test]
    fn test_wavelet_packet_node_cost() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let node = WaveletPacketNode::new(data, "test".to_string(), 1, 0);

        assert!(node.cost >= 0.0);
    }

    #[test]
    fn test_best_basis_criterion() {
        let wpt1 = WPT::new(WaveletType::Haar, 3).with_criterion(BestBasisCriterion::Shannon);
        assert_eq!(wpt1.criterion, BestBasisCriterion::Shannon);

        let wpt2 = WPT::new(WaveletType::Haar, 3).with_criterion(BestBasisCriterion::LogEnergy);
        assert_eq!(wpt2.criterion, BestBasisCriterion::LogEnergy);
    }
}
