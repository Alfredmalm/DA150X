from qiskit_aer.noise import NoiseModel, depolarizing_error

class QiskitNoiseModel:
    def __init__(self, single_qubit_error=0.01, two_qubit_error=0.01):
        """
        Initializes a noise model with specified depolarizing error rates.
        
        :param single_qubit_error: Probability of depolarizing error for single-qubit gates.
        :param two_qubit_error: Probability of depolarizing error for two-qubit gates.
        """
        self.noise_model = NoiseModel()
        
        # Define the depolarizing errors
        depol_error_1q = depolarizing_error(single_qubit_error, 1)
        depol_error_2q = depolarizing_error(two_qubit_error, 2)
        
        # Apply to all qubits
        self.noise_model.add_all_qubit_quantum_error(depol_error_1q, ['u1', 'u2', 'u3'])
        self.noise_model.add_all_qubit_quantum_error(depol_error_2q, ['cx'])
    
    def get_noise_model(self):
        """Returns the noise model instance."""
        return self.noise_model
