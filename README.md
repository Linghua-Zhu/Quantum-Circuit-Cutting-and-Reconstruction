# Scalable Quantum Circuit Analysis via Circuit Cutting and Reconstruction
This project presents a scalable approach to analyzing large-scale quantum circuits through circuit cutting and reconstruction techniques. It addresses the critical challenge of simulating and understanding complex quantum systems that are beyond the capabilities of traditional full quantum state simulation methods.

## Key Features

- Scalable analysis of large quantum circuits
- Adaptive circuit partitioning using community detection methods
- Novel reconstruction of global circuit properties from subcircuits
- Integration of quantum simulation with graph theory and information theory

## Main Components

- `scalable_circuit_cutter_and_reconstructor.py`: Main script for circuit analysis
- `schmidt_decomposition_and_merge.py`: Script for optimizing cross-subsystem quantum gate handling

## Significance

As quantum computing advances, the need for tools to analyze and optimize complex quantum algorithms grows. This project bridges the gap between theoretical quantum algorithm design and practical implementation constraints, contributing to the progress of quantum computing research and development.

## Advantages

1. **Scalability**: Handles circuits with a large number of qubits
2. **Flexibility**: Adapts to various circuit structures
3. **Computational Efficiency**: Reduces resource requirements through subcircuit analysis
4. **Global Insights**: Provides approximations of overall circuit behavior
5. **Comprehensive Output**: Offers multiple metrics for in-depth analysis

## Methodology

Our approach involves:
1. Graph-based circuit partitioning
2. Independent subcircuit analysis
3. Feature extraction from subcircuits
4. Approximate reconstruction of global circuit properties

## Major Challenges and Solutions

Initially, we attempted to use ‘Schmidt decomposition’ for truncating states across subcircuits, preserving significant singular values, followed by quantum state reconstruction using tensor networks. However, this approach encountered severe memory limitations when dealing with large-scale circuits, rendering the program inoperable.

To address this issue, we implemented a series of approximation measures:
- Utilization of graph partitioning methods to divide the circuit into smaller subcircuits.
- Independent analysis of each subcircuit to extract key features.
- Development of an approximation method (layered tensor networks) to combine subcircuit features and reconstruct global circuit properties.
- Employment of statistical quantities such as probability distributions and entropy to describe quantum states, rather than complete state vectors.

These approximation measures significantly reduced memory requirements, enabling the analysis of large-scale quantum circuits while still providing valuable insights.

## Areas for Future Improvement

1. Refinement of tensor decomposition and truncation methods
2. Optimization of cross-subsystem quantum gate handling techniques
   - Note: We have developed `schmidt_decomposition_and_merge.py` to address this, which is currently under testing and optimization
3. Improving accuracy of global state approximation
4. Exploring efficient memory management and distributed computing methods
5. Developing sophisticated partitioning strategies for specific circuit types
6. Investigating potential extensions to actual quantum hardware

## References

1. Orús, R. (2014). A Practical Introduction to Tensor Networks: Matrix Product States and Projected Entangled Pair States. Annals of Physics, 349, 117-158. [https://doi.org/10.1016/j.aop.2014.06.013](https://doi.org/10.1016/j.aop.2014.06.013)

2. Schollwöck, U. (2011). The Density-Matrix Renormalization Group in the Age of Matrix Product States. Annals of Physics, 326(1), 96-192. [https://doi.org/10.1016/j.aop.2010.09.012](https://doi.org/10.1016/j.aop.2010.09.012)

3. Peng, B., Harwood, S., Javadi-Abhari, A., & Gokhale, P. (2020). Simulating Large Quantum Circuits on Small Quantum Computers. Proceedings of the ACM International Conference on Computing Frontiers (CF'20), 1-10. [https://doi.org/10.1145/3387902.3392637](https://doi.org/10.1145/3387902.3392637)

4. Tang, H. L., Tomesh, T., Siraichi, Y., Javadi-Abhari, A., & Chuang, I. L. (2021). CutQC: Using Small Quantum Computers for Large Quantum Circuit Evaluations. Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '21), 473–486. [https://doi.org/10.1145/3445814.3446758](https://doi.org/10.1145/3445814.3446758)

5. Kan, Shuwen, et al. (2024). Scalable Circuit Cutting and Scheduling in a Resource-constrained and Distributed Quantum System. arXiv preprint arXiv:2405.04514.
