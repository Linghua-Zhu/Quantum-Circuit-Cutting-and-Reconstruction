import numpy as np
from qiskit.quantum_info import Statevector

def schmidt_decomposition_and_merge(state_a, state_b, threshold=1e-6):
    """
    Merge two subcircuit states using Schmidt decomposition.

    Args:
    state_a (Statevector): Statevector of subcircuit A.
    state_b (Statevector): Statevector of subcircuit B.
    threshold (float): Threshold to ignore smaller singular values.

    Returns:
    Statevector: Merged statevector.
    """
    # Tensor product of the two statevectors to combine them
    combined_state = state_a.tensor(state_b)
    
    # Reshape the combined state into matrix form
    dim_a = state_a.dim
    dim_b = state_b.dim
    reshaped_state = combined_state.data.reshape(dim_a, dim_b)
    
    # Perform SVD decomposition
    U, S, Vh = np.linalg.svd(reshaped_state, full_matrices=False)
    
    # Select significant singular values using the threshold
    significant_indices = S > threshold
    S_significant = S[significant_indices]
    U_significant = U[:, significant_indices]
    Vh_significant = Vh[significant_indices, :]
    
    # Reconstruct the approximate quantum state using significant singular values
    approx_state = np.zeros((dim_a, dim_b), dtype=complex)
    for i in range(len(S_significant)):
        approx_state += S_significant[i] * np.outer(U_significant[:, i], Vh_significant[i, :])
    
    # Flatten the approximate state back into a statevector
    approx_state_vector = approx_state.flatten()
    return Statevector(approx_state_vector)

def recursive_schmidt_merge(state_vectors, threshold=1e-6):
    """
    Recursively merge multiple subcircuit statevectors.

    Args:
    state_vectors (list of Statevector): List of statevectors for each subcircuit.
    threshold (float): Threshold to ignore smaller singular values.

    Returns:
    Statevector: Combined statevector of all subcircuits.
    """
    # Base case: if there is only one statevector, return it
    if len(state_vectors) == 1:
        return state_vectors[0]
    
    # Split the statevectors into two halves, merge them recursively
    mid = len(state_vectors) // 2
    left_merged = recursive_schmidt_merge(state_vectors[:mid], threshold)
    right_merged = recursive_schmidt_merge(state_vectors[mid:], threshold)
    
    # Use Schmidt decomposition to merge the left and right parts
    return schmidt_decomposition_and_merge(left_merged, right_merged, threshold)

