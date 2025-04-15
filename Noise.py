from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error, amplitude_damping_error

import numpy as np

class QiskitNoiseModelDepol:
    def __init__(self, single_qubit_error=0.01):
        """
        Initializes a noise model with specified depolarizing error rates.
        
        :param single_qubit_error: Probability of depolarizing error for single-qubit gates.
        :param two_qubit_error: Probability of depolarizing error for two-qubit gates.
        """
        self.noise_model = NoiseModel()
        
        # Define the depolarizing errors
        depol_error_1q = depolarizing_error(single_qubit_error, 1)
        
        # Apply to all qubits
        self.noise_model.add_all_qubit_quantum_error(depol_error_1q, ['u1', 'u2', 'u3'])

    
    def get_noise_model(self):
        """Returns the noise model instance."""
        return self.noise_model

class QiskitNoiseModelBitflip:
    def __init__(self, single_qubit_error=0.01):
        """
        Initializes a noise model with specified bit-flip error rates.
        
        Args:
            single_qubit_error (float): Probability of a bit-flip (X) error for single-qubit gates.
            two_qubit_error (float): Probability of a bit-flip (X⊗X) error for two-qubit gates.
        """
        self.noise_model = NoiseModel()
        
        # Define bit-flip errors
        bitflip_error_1q = pauli_error([('X', single_qubit_error), ('I', 1 - single_qubit_error)])
        
        # Apply to all single-qubit and two-qubit gates
        self.noise_model.add_all_qubit_quantum_error(bitflip_error_1q, ['u1', 'u2', 'u3'])
    
    def get_noise_model(self):
        """Returns the noise model instance."""
        return self.noise_model

class QiskitNoiseModelAmplitudeDamp:
    def __init__(self, single_qubit_error=0.01 ):
        """
        Initializes an amplitude damping noise model for any number of qubits.
        
        Args:
            t1 (float): T₁ relaxation time (in microseconds).
            gate_time (float): Duration of gates (in microseconds).
        """
        self.noise_model = NoiseModel()
        
        
        # Amplitude damping error for single-qubit gates
        amp_damping_error_1q = amplitude_damping_error( single_qubit_error, excited_state_population=0, canonical_kraus=True)
        

        
        # Apply to all single-qubit gates
        self.noise_model.add_all_qubit_quantum_error(amp_damping_error_1q, ['u1', 'u2', 'u3'])
    
    def get_noise_model(self):
        """Returns the noise model instance."""
        return self.noise_model