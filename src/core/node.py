"""
Basic implementation of a concept node in the node-based neural network architecture.
"""
import numpy as np
from typing import Dict, List, Optional, Set, Tuple


class Node:
    """
    Represents a concept node in the network.
    
    Each node can:
    - Store a state vector representing its conceptual content
    - Connect to other nodes with weighted connections
    - Activate based on input and propagate to connected nodes
    - Update its state through learning
    """
    
    def __init__(
        self,
        node_id: str,
        dimension: int = 128,
        activation_threshold: float = 0.5
    ):
        """
        Initialize a concept node.
        
        Args:
            node_id: Unique identifier for the node
            dimension: Dimensionality of the state vector
            activation_threshold: Threshold for node activation
        """
        self.node_id = node_id
        self.state = np.zeros(dimension)
        self.connections: Dict[str, Tuple[Node, float]] = {}  # node_id -> (node, weight)
        self.activation_level = 0.0
        self.activation_threshold = activation_threshold
        self.is_active = False
        
    def connect(self, target_node: 'Node', weight: float = 1.0):
        """
        Create a connection to another node.
        
        Args:
            target_node: The node to connect to
            weight: Connection strength
        """
        self.connections[target_node.node_id] = (target_node, weight)
        
    def activate(self, input_signal: np.ndarray) -> float:
        """
        Activate the node based on input signal.
        
        Args:
            input_signal: Input vector to the node
            
        Returns:
            Activation level (0.0 to 1.0)
        """
        # Simple activation function: cosine similarity between input and state
        if np.any(self.state) and np.any(input_signal):
            similarity = np.dot(self.state, input_signal) / (
                np.linalg.norm(self.state) * np.linalg.norm(input_signal)
            )
            # Scale to 0-1 range
            self.activation_level = (similarity + 1) / 2
        else:
            self.activation_level = 0.0
            
        self.is_active = self.activation_level >= self.activation_threshold
        return self.activation_level
    
    def propagate(self) -> List['Node']:
        """
        Propagate activation to connected nodes if this node is active.
        
        Returns:
            List of nodes that were activated
        """
        if not self.is_active:
            return []
            
        activated_nodes = []
        for node_id, (node, weight) in self.connections.items():
            # Propagate weighted activation to connected node
            input_signal = self.state * self.activation_level * weight
            node_activation = node.activate(input_signal)
            
            if node.is_active:
                activated_nodes.append(node)
                
        return activated_nodes
    
    def update_state(self, input_data: np.ndarray, learning_rate: float = 0.1):
        """
        Update node state based on new input data.
        
        Args:
            input_data: New data to incorporate
            learning_rate: Rate of state update
        """
        # Simple update rule: move state vector towards input data
        if np.any(input_data):
            normalized_input = input_data / np.linalg.norm(input_data)
            self.state = (1 - learning_rate) * self.state + learning_rate * normalized_input
            # Normalize state vector
            if np.any(self.state):
                self.state = self.state / np.linalg.norm(self.state)
    
    def __repr__(self) -> str:
        return f"Node(id={self.node_id}, active={self.is_active}, " \
               f"activation={self.activation_level:.2f}, connections={len(self.connections)})"