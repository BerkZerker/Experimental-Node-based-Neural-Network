"""
Tests for the Node class.
"""
import unittest
import numpy as np
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.node import Node


class TestNode(unittest.TestCase):
    """Test cases for the Node class."""
    
    def test_node_initialization(self):
        """Test that a node initializes with correct default values."""
        node = Node("test_node")
        self.assertEqual(node.node_id, "test_node")
        self.assertEqual(node.activation_level, 0.0)
        self.assertFalse(node.is_active)
        self.assertEqual(len(node.connections), 0)
        self.assertEqual(node.state.shape, (128,))  # Default dimension
        
    def test_node_connection(self):
        """Test connection between nodes."""
        node1 = Node("node1")
        node2 = Node("node2")
        
        node1.connect(node2, weight=0.5)
        
        self.assertEqual(len(node1.connections), 1)
        self.assertIn(node2.node_id, node1.connections)
        self.assertEqual(node1.connections[node2.node_id][0], node2)
        self.assertEqual(node1.connections[node2.node_id][1], 0.5)
        
    def test_node_activation(self):
        """Test node activation based on input signal."""
        node = Node("test_node", activation_threshold=0.6)
        
        # Set a state vector
        node.state = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Test with aligned input (should activate)
        input_signal = np.array([0.9, 0.1, 0.0, 0.0])
        activation_level = node.activate(input_signal)
        
        self.assertGreaterEqual(activation_level, 0.6)
        self.assertTrue(node.is_active)
        
        # Test with orthogonal input (should not activate)
        input_signal = np.array([0.0, 1.0, 0.0, 0.0])
        activation_level = node.activate(input_signal)
        
        self.assertLess(activation_level, 0.6)
        self.assertFalse(node.is_active)
        
    def test_propagation(self):
        """Test activation propagation to connected nodes."""
        node1 = Node("node1", dimension=4)
        node2 = Node("node2", dimension=4)
        node3 = Node("node3", dimension=4)
        
        # Connect node1 to node2 and node3
        node1.connect(node2)
        node1.connect(node3)
        
        # Set states
        node1.state = np.array([1.0, 0.0, 0.0, 0.0])
        node2.state = np.array([0.9, 0.1, 0.0, 0.0])  # Similar to node1
        node3.state = np.array([0.0, 1.0, 0.0, 0.0])  # Different from node1
        
        # Activate node1
        node1.activate(np.array([1.0, 0.0, 0.0, 0.0]))
        
        # Propagate activation
        activated_nodes = node1.propagate()
        
        # node2 should be activated, node3 should not
        self.assertEqual(len(activated_nodes), 1)
        self.assertIn(node2, activated_nodes)
        self.assertNotIn(node3, activated_nodes)
        
    def test_state_update(self):
        """Test updating node state with new data."""
        node = Node("test_node", dimension=2)
        
        # Initial state
        self.assertTrue(np.allclose(node.state, np.array([0.0, 0.0])))
        
        # Update with new data
        node.update_state(np.array([1.0, 0.0]), learning_rate=0.5)
        
        # State should be updated and normalized
        expected_state = np.array([1.0, 0.0])  # Normalized
        self.assertTrue(np.allclose(node.state, expected_state))
        
        # Update with more data
        node.update_state(np.array([1.0, 1.0]), learning_rate=0.2)
        
        # State should move toward the new data
        # (0.8 * [1, 0] + 0.2 * [1, 1]) / norm
        expected_direction = 0.8 * np.array([1.0, 0.0]) + 0.2 * np.array([1.0, 1.0])
        expected_state = expected_direction / np.linalg.norm(expected_direction)
        self.assertTrue(np.allclose(node.state, expected_state))


if __name__ == "__main__":
    unittest.main()