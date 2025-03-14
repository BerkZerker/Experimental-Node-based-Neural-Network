"""
A simple experiment to demonstrate node activation and propagation.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

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


def visualize_activation(nodes, active_nodes):
    """Visualize which nodes are active."""
    plt.figure(figsize=(8, 6))
    
    # Create positions for the nodes
    positions = {
        "cat": (0, 0),
        "pet": (1, 0),
        "dog": (2, 0),
        "animal": (1, 1)
    }
    
    # Draw connections
    for node_id, node in nodes.items():
        for conn_id in node.connections:
            conn_node = nodes.get(conn_id)
            if conn_node:
                plt.plot(
                    [positions[node_id][0], positions[conn_id][0]],
                    [positions[node_id][1], positions[conn_id][1]],
                    'k-', alpha=0.3
                )
    
    # Draw nodes
    for node_id, node in nodes.items():
        color = 'r' if node_id in active_nodes else 'b'
        alpha = 1.0 if node_id in active_nodes else 0.5
        plt.scatter(
            positions[node_id][0],
            positions[node_id][1],
            s=500,
            color=color,
            alpha=alpha
        )
        plt.text(
            positions[node_id][0],
            positions[node_id][1],
            node_id,
            ha='center',
            va='center',
            color='white'
        )
    
    plt.title("Node Activation Visualization")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("node_activation.png")
    plt.close()


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
    
    # Visualize the network
    print("\nGenerating visualization...")
    all_active = active_nodes + propagated_nodes
    visualize_activation(nodes, all_active)
    print("Visualization saved as 'node_activation.png'")


if __name__ == "__main__":
    main()