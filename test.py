import qiskit
from qiskit_aer.primitives import SamplerV2
from qiskit import execute
from qiskit.quantum_info import SparsePauliOp
 
create_query = lambda N, k: SparsePauliOp.from_list([("Z" * N, 1), ("Z" * (N - k) + "X" + "Z" * k, -1)])  # Create the query operator


def Z(n, A):
    qc = QuantumCircuit(n)
    qc.unitary(A, range(n), label='Z')
    return qc

def R(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)
    qc.x(range(n))
    qc.h(range(n))
    return qc

def grover_iteration(n, A):
    G = QuantumCircuit(n)  # Instantiate a quantum circuit of n qubits
    G.append(Z(n, A))  # Append the appropriate phase inversion Z
    G.append(R(n))  # Append the diffusion operator R for n qubits
    return G

def grover_experiment(n, k, m):
    N = 2 ** n
    A = create_query(N, k)  # Query k of the N possible states
    qc = QuantumCircuit(n)  # Instantiate a quantum circuit of n qubits
    G = grover_iteration(n, A)  # Create the circuit of a single Grover iteration
    qc.append(qc.h(n))  # Append initial Hadamard transform
    for i in range(m):  # Append the correct number of Grover iterations
        qc.append(G)
    qc.measure_all()  # Append terminating measurement gates
    successes = []
    for i in range(100):  # Simulate the circuit
        s = simulate_circuit(qc, 10000)
        successes.append(s)  # Collect the test data
    return successes


def simulate_circuit(qc, shots):
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend=simulator, shots=shots).result()
    counts = result.get_counts(qc)
    return counts




import matplotlib.pyplot as plt

def plot_results(successes):
    # Aggregate results
    aggregated_counts = {}
    for result in successes:
        for key, value in result.items():
            if key in aggregated_counts:
                aggregated_counts[key] += value
            else:
                aggregated_counts[key] = value

    # Plot results
    plt.bar(aggregated_counts.keys(), aggregated_counts.values())
    plt.xlabel('States')
    plt.ylabel('Counts')
    plt.title('Grover Experiment Results')
    plt.show()

def main():
    N = 4
    k = 2
    m = 1
    successes = grover_experiment(N, k, m)
    plot_results(successes)