import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from networkx.algorithms import community

def analyze_circuit(circuit):
    """
    Analyze the quantum circuit to extract basic information and connections.
    
    Args:
    circuit (QuantumCircuit): The quantum circuit to analyze.
    
    Returns:
    tuple: (number of qubits, number of gates, list of CZ/CNOT connections)
    """
    n_qubits = circuit.num_qubits
    n_gates = len(circuit)
    cz_cnot_connections = []
    for inst in circuit.data:
        if inst.operation.name in ['cz', 'cx']:
            control, target = [q._index for q in inst.qubits]
            cz_cnot_connections.append((control, target))
    return n_qubits, n_gates, cz_cnot_connections

def circuit_to_graph(n_qubits, connections):
    """
    Convert the circuit connections to a graph representation.
    
    Args:
    n_qubits (int): Number of qubits in the circuit.
    connections (list): List of CZ/CNOT connections.
    
    Returns:
    nx.Graph: Graph representation of the circuit.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_qubits))
    
    # Add all possible edges with default weight 0.0
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            G.add_edge(i, j, weight=0.0)
    
    # Set weight to 1.0 for CZ or CNOT connections
    for connection in connections:
        G[connection[0]][connection[1]]['weight'] = 1.0
    
    return G

def partition_graph(graph, num_partitions=5):
    """
    Partition the graph into subgraphs using the Louvain method.
    
    Args:
    graph (nx.Graph): The graph to partition.
    num_partitions (int): The desired number of partitions.
    
    Returns:
    list: List of partitions, where each partition is a list of node indices.
    """
    partitions = []
    remaining_graph = graph.copy()

    while len(partitions) < num_partitions - 1:
        communities = community.louvain_communities(remaining_graph, seed=42)
        largest_community = max(communities, key=len)
        partitions.append(list(largest_community))
        remaining_graph.remove_nodes_from(largest_community)

    partitions.append(list(remaining_graph.nodes()))

    min_partition_size = len(graph.nodes()) // (num_partitions * 2)
    final_partitions = []
    for partition in partitions:
        if len(partition) < min_partition_size and final_partitions:
            closest_partition = min(final_partitions, key=lambda x: nx.shortest_path_length(graph, partition[0], x[0]))
            closest_partition.extend(partition)
        else:
            final_partitions.append(partition)

    while len(final_partitions) < num_partitions:
        largest_partition = max(final_partitions, key=len)
        final_partitions.remove(largest_partition)
        mid = len(largest_partition) // 2
        final_partitions.append(largest_partition[:mid])
        final_partitions.append(largest_partition[mid:])

    return final_partitions

def create_subcircuits(circuit, partitions):
    """
    Create subcircuits based on the partitions.
    
    Args:
    circuit (QuantumCircuit): The original quantum circuit.
    partitions (list): List of partitions.
    
    Returns:
    list: List of subcircuits (QuantumCircuit objects).
    """
    subcircuits = []
    for part in partitions:
        subcircuit = QuantumCircuit(len(part))
        qubit_map = {old: new for new, old in enumerate(sorted(part))}
        for inst in circuit.data:
            if all(q._index in part for q in inst.qubits):
                new_qubits = [qubit_map[q._index] for q in inst.qubits]
                if inst.operation.name != 'measure':
                    subcircuit.append(inst.operation, new_qubits)
        subcircuits.append(subcircuit)
    return subcircuits

def execute_circuit_statevector(circuit):
    """
    Execute the circuit and obtain its statevector.
    
    Args:
    circuit (QuantumCircuit): The quantum circuit to execute.
    
    Returns:
    np.array: The statevector of the circuit.
    """
    backend = Aer.get_backend('statevector_simulator')
    job = backend.run(circuit)
    result = job.result()
    statevector = result.get_statevector()
    return statevector

def extract_features(statevector, num_features=20):
    """
    Extract features from the statevector.
    
    Args:
    statevector (np.array): The statevector of the circuit.
    num_features (int): Number of top features to extract.
    
    Returns:
    tuple: (top indices, top probabilities, entropy, top states)
    """
    probabilities = np.abs(statevector.data)**2
    top_indices = np.argsort(probabilities)[-num_features:]
    top_probabilities = probabilities[top_indices]
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    top_states = [format(idx, f'0{int(np.log2(statevector.dim))}b') for idx in top_indices]
    
    return top_indices, top_probabilities, entropy, top_states

def combine_features(features_list, partitions, n_qubits):
    """
    Combine features from all subcircuits.
    
    Args:
    features_list (list): List of features from each subcircuit.
    partitions (list): List of partitions.
    n_qubits (int): Total number of qubits in the original circuit.
    
    Returns:
    tuple: (combined indices, combined probabilities, combined entropy, approximate bitstring, top states)
    """
    combined_indices = np.concatenate([f[0] for f in features_list])
    combined_probabilities = np.concatenate([f[1] for f in features_list])
    combined_entropy = np.mean([f[2] for f in features_list])
    combined_probabilities /= np.sum(combined_probabilities)
    
    full_bitstring = ['0'] * n_qubits
    
    for partition, features in zip(partitions, features_list):
        most_probable_state = features[3][-1]
        for i, qubit in enumerate(sorted(partition)):
            full_bitstring[qubit] = most_probable_state[i]
    
    combined_bitstring = ''.join(full_bitstring)
    
    top_states = []
    for i in range(min(100, len(combined_indices))):
        state = format(combined_indices[i], f'0{n_qubits}b')
        prob = combined_probabilities[i]
        top_states.append((state, prob))
    
    return combined_indices, combined_probabilities, combined_entropy, combined_bitstring, top_states

# Main program
circuit_path = "/Users/zhulinghua/Documents/Bluequbits/circuit_2_42q.qasm"
circuit = QuantumCircuit.from_qasm_file(circuit_path)

# Analyze the circuit
n_qubits, n_gates, connections = analyze_circuit(circuit)
print(f"Qubits: {n_qubits}, Gates: {n_gates}")
print(f"CZ/CNOT connections: {connections[:5]}...")

# Convert circuit to graph
G = circuit_to_graph(n_qubits, connections)

# Partition the graph
partitions = partition_graph(G, num_partitions=5)
print(f"Number of partitions: {len(partitions)}")
for i, part in enumerate(partitions):
    print(f"Partition {i} size: {len(part)}")

# Create subcircuits
subcircuits = create_subcircuits(circuit, partitions)

# Process each subcircuit
features_list = []
for i, subcircuit in enumerate(subcircuits):
    print(f"\nProcessing subcircuit {i} with {subcircuit.num_qubits} qubits...")
    statevector = execute_circuit_statevector(subcircuit)
    print(f"Statevector for subcircuit {i} obtained. Shape: {statevector.dim}")
    features = extract_features(statevector)
    features_list.append(features)
    print(f"Features extracted for subcircuit {i}")

# Combine features
combined_indices, combined_probabilities, combined_entropy, approximate_bitstring, top_states = combine_features(features_list, partitions, n_qubits)

# Print results
print("\nApproximate state summary:")
print(f"Number of significant states: {len(combined_indices)}")
print(f"Top 5 states and their probabilities:")
for state, prob in top_states[:5]:
    print(f"  State {state}: {prob:.4f}")
print(f"Approximate entropy: {combined_entropy:.4f}")
print(f"Most probable {n_qubits}-qubit bitstring: {approximate_bitstring}")

print(f"\nTop {len(top_states)} most probable states:")
for i, (state, prob) in enumerate(top_states, 1):
    print(f"{i}. State: {state}, Probability: {prob:.6f}")

print("\nAnalysis complete.")