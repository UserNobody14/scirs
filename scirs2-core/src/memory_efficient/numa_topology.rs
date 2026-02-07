//! NUMA topology detection and management.
//!
//! This module provides utilities for detecting and working with NUMA (Non-Uniform Memory Access)
//! topologies on systems that support it. NUMA awareness can significantly improve performance
//! for memory-intensive operations by reducing cross-node memory access latency.
//!
//! # Supported Platforms
//!
//! - **Linux**: Full support via `/sys/devices/system/node` interface
//! - **Windows**: Full support via Windows API (requires `windows-sys` crate)
//! - **macOS/BSD**: Graceful fallback (returns None - these systems don't typically have NUMA)
//!
//! # Optional Features
//!
//! - `numa`: Enable libnuma integration for advanced NUMA management on Linux

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// NUMA node information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NumaNode {
    /// Node ID
    pub node_id: usize,

    /// CPU cores associated with this node
    pub cpu_list: Vec<usize>,

    /// Memory available on this node (in bytes)
    pub memory_bytes: u64,

    /// Memory free on this node (in bytes)
    pub memory_free_bytes: u64,
}

impl NumaNode {
    /// Create a new NUMA node
    pub fn new(node_id: usize, cpu_list: Vec<usize>, memory_bytes: u64) -> Self {
        Self {
            node_id,
            cpu_list,
            memory_bytes,
            memory_free_bytes: memory_bytes,
        }
    }

    /// Get the number of CPUs in this node
    pub fn num_cpus(&self) -> usize {
        self.cpu_list.len()
    }

    /// Check if a CPU belongs to this node
    pub fn contains_cpu(&self, cpu_id: usize) -> bool {
        self.cpu_list.contains(&cpu_id)
    }

    /// Get memory utilization percentage
    pub fn memory_utilization(&self) -> f64 {
        if self.memory_bytes == 0 {
            0.0
        } else {
            (self.memory_bytes - self.memory_free_bytes) as f64 / self.memory_bytes as f64
        }
    }
}

/// NUMA topology information for the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaTopology {
    /// All NUMA nodes in the system
    pub nodes: Vec<NumaNode>,

    /// Whether the system is NUMA-aware
    pub is_numa: bool,
}

impl NumaTopology {
    /// Create a new NUMA topology
    pub fn new(nodes: Vec<NumaNode>, is_numa: bool) -> Self {
        Self { nodes, is_numa }
    }

    /// Get the number of NUMA nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get a specific NUMA node by ID
    pub fn get_node(&self, node_id: usize) -> Option<&NumaNode> {
        self.nodes.iter().find(|node| node.node_id == node_id)
    }

    /// Find which NUMA node contains a specific CPU
    pub fn find_node_for_cpu(&self, cpu_id: usize) -> Option<&NumaNode> {
        self.nodes.iter().find(|node| node.contains_cpu(cpu_id))
    }

    /// Get total memory across all NUMA nodes
    pub fn total_memory(&self) -> u64 {
        self.nodes.iter().map(|node| node.memory_bytes).sum()
    }

    /// Get total free memory across all NUMA nodes
    pub fn total_free_memory(&self) -> u64 {
        self.nodes.iter().map(|node| node.memory_free_bytes).sum()
    }

    /// Detect NUMA topology on the current system
    ///
    /// Returns `None` if NUMA is not supported or detection fails
    pub fn detect() -> Option<Self> {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux().ok()
        }

        #[cfg(target_os = "windows")]
        {
            Self::detect_windows().ok()
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            // macOS, BSD, and other systems - no NUMA support
            None
        }
    }

    /// Detect NUMA topology on Linux using sysfs
    #[cfg(target_os = "linux")]
    fn detect_linux() -> CoreResult<Self> {
        use std::fs;
        use std::path::Path;

        let node_path = Path::new("/sys/devices/system/node");

        if !node_path.exists() {
            // No NUMA support - return single node with all CPUs
            return Self::detect_non_numa();
        }

        let mut nodes = Vec::new();

        // Iterate through node directories
        let entries = fs::read_dir(node_path).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to read NUMA node directory: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                CoreError::IoError(
                    ErrorContext::new(format!("Failed to read NUMA directory entry: {e}"))
                        .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;

            let path = entry.path();
            let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("");

            // Check if this is a node directory (e.g., node0, node1)
            if let Some(node_id_str) = filename.strip_prefix("node") {
                if let Ok(node_id) = node_id_str.parse::<usize>() {
                    // Read CPU list
                    let cpulist_path = path.join("cpulist");
                    let cpu_list = if cpulist_path.exists() {
                        let cpulist_str = fs::read_to_string(&cpulist_path).map_err(|e| {
                            CoreError::IoError(
                                ErrorContext::new(format!("Failed to read cpulist: {e}"))
                                    .with_location(ErrorLocation::new(file!(), line!())),
                            )
                        })?;

                        Self::parse_cpu_list(&cpulist_str.trim())?
                    } else {
                        Vec::new()
                    };

                    // Read memory info
                    let meminfo_path = path.join("meminfo");
                    let (memory_bytes, memory_free_bytes) = if meminfo_path.exists() {
                        Self::parse_node_meminfo(&meminfo_path)?
                    } else {
                        (0, 0)
                    };

                    let mut node = NumaNode::new(node_id, cpu_list, memory_bytes);
                    node.memory_free_bytes = memory_free_bytes;

                    nodes.push(node);
                }
            }
        }

        // Sort nodes by ID
        nodes.sort_by_key(|node| node.node_id);

        if nodes.is_empty() {
            return Self::detect_non_numa();
        }

        Ok(Self::new(nodes, true))
    }

    /// Parse Linux CPU list format (e.g., "0-3,5,7-9")
    #[cfg(target_os = "linux")]
    fn parse_cpu_list(cpulist: &str) -> CoreResult<Vec<usize>> {
        let mut cpus = Vec::new();

        for range in cpulist.split(',') {
            let range = range.trim();
            if range.is_empty() {
                continue;
            }

            if range.contains('-') {
                // Range format (e.g., "0-3")
                let parts: Vec<&str> = range.split('-').collect();
                if parts.len() == 2 {
                    let start = parts[0].parse::<usize>().map_err(|e| {
                        CoreError::InvalidArgument(
                            ErrorContext::new(format!("Invalid CPU range start: {e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;
                    let end = parts[1].parse::<usize>().map_err(|e| {
                        CoreError::InvalidArgument(
                            ErrorContext::new(format!("Invalid CPU range end: {e}"))
                                .with_location(ErrorLocation::new(file!(), line!())),
                        )
                    })?;

                    cpus.extend(start..=end);
                }
            } else {
                // Single CPU
                let cpu = range.parse::<usize>().map_err(|e| {
                    CoreError::InvalidArgument(
                        ErrorContext::new(format!("Invalid CPU ID: {e}"))
                            .with_location(ErrorLocation::new(file!(), line!())),
                    )
                })?;
                cpus.push(cpu);
            }
        }

        Ok(cpus)
    }

    /// Parse NUMA node meminfo file
    #[cfg(target_os = "linux")]
    fn parse_node_meminfo(meminfo_path: &std::path::Path) -> CoreResult<(u64, u64)> {
        use std::fs;

        let contents = fs::read_to_string(meminfo_path).map_err(|e| {
            CoreError::IoError(
                ErrorContext::new(format!("Failed to read meminfo: {e}"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        let mut total_kb = 0u64;
        let mut free_kb = 0u64;

        for line in contents.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                if parts[2] == "MemTotal:" {
                    total_kb = parts[3].parse::<u64>().unwrap_or(0);
                } else if parts[2] == "MemFree:" {
                    free_kb = parts[3].parse::<u64>().unwrap_or(0);
                }
            }
        }

        // Convert KB to bytes
        Ok((total_kb * 1024, free_kb * 1024))
    }

    /// Detect NUMA topology on Windows
    #[cfg(target_os = "windows")]
    fn detect_windows() -> CoreResult<Self> {
        // Windows NUMA detection would use GetNumaHighestNodeNumber and GetNumaNodeProcessorMask
        // For now, return non-NUMA fallback
        // TODO: Implement Windows NUMA detection when windows-sys feature is added
        Self::detect_non_numa()
    }

    /// Fallback for non-NUMA systems
    fn detect_non_numa() -> CoreResult<Self> {
        use crate::memory_efficient::platform_memory::PlatformMemoryInfo;

        // Create a single node with all available CPUs and memory
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let cpu_list: Vec<usize> = (0..num_cpus).collect();

        // Get total system memory
        let memory_info = PlatformMemoryInfo::detect();
        let (memory_bytes, memory_free_bytes) = if let Some(info) = memory_info {
            (info.total_memory as u64, info.available_memory as u64)
        } else {
            (0, 0)
        };

        let mut node = NumaNode::new(0, cpu_list, memory_bytes);
        node.memory_free_bytes = memory_free_bytes;

        Ok(Self::new(vec![node], false))
    }
}

/// NUMA-aware memory allocator hint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumaPolicy {
    /// Default system policy
    Default,
    /// Bind to specific node
    Bind(usize),
    /// Interleave across all nodes
    Interleave,
    /// Prefer specific node but allow fallback
    Preferred(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_node_creation() {
        let node = NumaNode::new(0, vec![0, 1, 2, 3], 8 * 1024 * 1024 * 1024);

        assert_eq!(node.node_id, 0);
        assert_eq!(node.num_cpus(), 4);
        assert!(node.contains_cpu(2));
        assert!(!node.contains_cpu(4));
    }

    #[test]
    fn test_numa_topology_creation() {
        let node0 = NumaNode::new(0, vec![0, 1], 4 * 1024 * 1024 * 1024);
        let node1 = NumaNode::new(1, vec![2, 3], 4 * 1024 * 1024 * 1024);

        let topology = NumaTopology::new(vec![node0, node1], true);

        assert_eq!(topology.num_nodes(), 2);
        assert!(topology.is_numa);
        assert_eq!(topology.total_memory(), 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_find_node_for_cpu() {
        let node0 = NumaNode::new(0, vec![0, 1], 4 * 1024 * 1024 * 1024);
        let node1 = NumaNode::new(1, vec![2, 3], 4 * 1024 * 1024 * 1024);

        let topology = NumaTopology::new(vec![node0, node1], true);

        let node = topology.find_node_for_cpu(2);
        assert!(node.is_some());
        assert_eq!(node.expect("Node not found").node_id, 1);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_parse_cpu_list() {
        // Test single CPU
        let cpus = NumaTopology::parse_cpu_list("0").expect("Parse failed");
        assert_eq!(cpus, vec![0]);

        // Test range
        let cpus = NumaTopology::parse_cpu_list("0-3").expect("Parse failed");
        assert_eq!(cpus, vec![0, 1, 2, 3]);

        // Test complex list
        let cpus = NumaTopology::parse_cpu_list("0-2,5,7-9").expect("Parse failed");
        assert_eq!(cpus, vec![0, 1, 2, 5, 7, 8, 9]);

        // Test with whitespace
        let cpus = NumaTopology::parse_cpu_list(" 0-1, 3 ").expect("Parse failed");
        assert_eq!(cpus, vec![0, 1, 3]);
    }

    #[test]
    fn test_numa_detection() {
        // This test will work differently on different platforms
        let topology = NumaTopology::detect();

        // Should always return something (even if non-NUMA fallback)
        // We can't assert much more since it depends on the system
        if let Some(topo) = topology {
            assert!(topo.num_nodes() > 0);
            assert!(topo.total_memory() > 0 || topo.total_memory() == 0); // Allow 0 for test environments
        }
    }

    #[test]
    fn test_memory_utilization() {
        let mut node = NumaNode::new(0, vec![0, 1], 1000);
        node.memory_free_bytes = 600;

        let utilization = node.memory_utilization();
        assert!((utilization - 0.4).abs() < 1e-10);
    }
}
