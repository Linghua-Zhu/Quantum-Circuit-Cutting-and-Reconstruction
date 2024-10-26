from qiskit import QuantumCircuit
import bluequbit
import matplotlib.pyplot as plt
from collections import Counter

bq_client = bluequbit.init("3l1f2zZi7qJjO7o9lFY7KmLaBXxVq6fi")

# Read circuits
circuit_1_path = "/Users/zhulinghua/Documents/Bluequbits/circuit_1_30q_fixed.qasm"  
circuit_1 = QuantumCircuit.from_qasm_file(circuit_1_path)

print("Number of qubits:", circuit_1.num_qubits)
print("Circuit depth:", circuit_1.depth())

# Estimate cost and runtime
#estimate_result = bq_client.estimate(circuit_1, device='cpu')
#print("Estimated cost and runtime (CPU):", estimate_result)

# Run the circuit on the BlueQubit simulator (we can choose GPU, CPU or real quantum devices)
job_result_1 = bq_client.run(circuit_1, device='gpu', shots=1024, job_name="gpu_circuit_1")
counts = job_result_1.get_counts()
print("Measurement results:", counts)

# Find the highest probability bitstring
highest_prob_bitstring = max(counts, key=counts.get)
print("The highest probability bitstring is:", highest_prob_bitstring)

# Visualize measurement results with a bar chart
# Sorting results for better visualization
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:30] 

# Extract bitstrings and counts
bitstrings = [item[0] for item in sorted_counts]
frequencies = [item[1] for item in sorted_counts]

# Plotting the bar chart for the measurement results
plt.figure(figsize=(12, 6))
plt.bar(bitstrings, frequencies, color='lightblue')
plt.xlabel('Bitstrings')
plt.ylabel('Number of Occurrences')
plt.xticks(rotation=90, fontsize=8)
plt.title('Top 30 Most Frequent Measured Bitstrings')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

