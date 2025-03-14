"""
A simple experiment to demonstrate node activation and propagation.
"""
import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.node import Node


def create_simple_network():
    """Create a simple network of connected nodes."""
    # Create nodes for different concepts
    cat_node = Node("cat", dimension=4)
    pet_node = Node("pet", dimension=4)
    dog_node = Node("dog", dimension=4)
    animal_node = Node("animal", dimension=4)
    
    # Set conceptual states
    cat_node.state = np.array([0.8, 0.6, 0.0, 0.0])
    pet_node.state = np.array([0.5, 0.8, 0.2, 0.0])
    dog_node.state = np.array([0.7, 0.5, 0.3, 0.0])
    animal_node.state = np.array([0.9, 0.2, 0.1, 0.3])
    
    # Normalize states
    for node in [cat_node, pet_node, dog_node, animal_node]:
        node.state = node.state / np.linalg.norm(node.state)
    
    # Create connections
    cat_node.connect(pet_node, weight=0.8)
    cat_node.connect(animal_node, weight=0.9)
    dog_node.connect(pet_node, weight=0.8)
    dog_node.connect(animal_node, weight=0.9)
    pet_node.connect(animal_node, weight=0.7)
    
    return {
        "cat": cat_node,
        "pet": pet_node,
        "dog": dog_node,
        "animal": animal_node
    }


def print_network_state(nodes, active_nodes):
    """Print a text-based visualization of the network."""
    print("\nNetwork State:")
    print("-------------")
    
    # Print nodes and their activation status
    for node_id, node in nodes.items():
        status = "ACTIVE" if node_id in active_nodes else "inactive"
        connections = ", ".join(node.connections.keys())
        print(f"Node: {node_id.upper()} [{status}]")
        print(f"  - Activation: {node.activation_level:.4f}")
        print(f"  - Connections: {connections}")
    
    print("\nActive Path:")
    print("-----------")
    
    # Show the activation path
    if active_nodes:
        path = " -> ".join(active_nodes)
        print(f"{path}")
    else:
        print("No nodes activated")


def main():
    """Run the activation experiment."""
    print("Creating a simple concept network...")
    nodes = create_simple_network()
    
    # Create an input related to 'cat'
    cat_input = np.array([0.8, 0.6, 0.1, 0.0])
    cat_input = cat_input / np.linalg.norm(cat_input)
    
    print("Input: Vector similar to 'cat' concept")
    
    # Activate nodes based on input
    activation_levels = {}
    for node_id, node in nodes.items():
        activation_levels[node_id] = node.activate(cat_input)
        print(f"Node '{node_id}' activation: {activation_levels[node_id]:.4f}")
    
    # Get active nodes
    active_nodes = [node_id for node_id, node in nodes.items() if node.is_active]
    print(f"Active nodes: {active_nodes}")
    
    # Propagate activation
    print("\nPropagating activation from active nodes...")
    propagated_nodes = []
    for node_id in active_nodes:
        activated = nodes[node_id].propagate()
        for node in activated:
            if node.node_id not in propagated_nodes and node.node_id not in active_nodes:
                propagated_nodes.append(node.node_id)
    
    print(f"Additionally activated nodes after propagation: {propagated_nodes}")
    
    # Print network state
    all_active = active_nodes + propagated_nodes
    print_network_state(nodes, all_active)
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()