# 
# This code is part of Qton.
# Qton Version: 2.0.0
# 
# File:   superoperator.py
# Author: Yunheng Ma
# Date :  2022-01-27
#


import numpy as np


__all__ = ["Qsuperoperator"]


# alphabet
alp = 'abcdefghijklmnopqrstuvwxyz'
ALP = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


from ._basic_qcircuit_ import _Basic_qcircuit_
from .density_matrix import Qdensity_matrix
from qton.operators.superop import to_superop


class Qsuperoperator(Qdensity_matrix):
    '''
    Quantum circuit operated by super operators.

    This is a wrapped of "Qdensity_matrix".

    -Attributes(3):
        1. backend --- how to execute the circuit; 'statevector', 
            'density_matrix', 'unitary', or 'superoperator'.
            type: str
        2. num_qubits --- number of qubits.
            type: int
        3. state --- circuit state representation.
            type: numpy.ndarray, complex        
    
    -Methods(72):
        1. __init__(self, num_qubits=1)
        2. _apply_1q_(self, op, qubit)
        3. _apply_2q_(self, op, qubit1, qubit2)
        4. _apply_3q_(self, op, qubit1, qubit2, qubit3)
        5. apply_1q(self, op, qubits)
        6. apply_2q(self, op, qubits1, qubits2)
        7. apply_3q(self, op, qubits1, qubits2, qubits3)
        8. apply(self, op, *qubits)
        9. measure(self, qubit, delete=False)
        10. add_qubit(self, num_qubits=1)
        11. sample(self, shots=1024, output='memory')
        12. reduce(self, qubit)
        13. bit_flip(self, p, qubits)
        14. phase_flip(self, p, qubits)
        15. bit_phase_flip(self, p, qubits)
        16. depolarize(self, p, qubits)
        17. amplitude_damping(self, gamma, qubits)
        18. generalized_amplitude_damping(self, p, gamma, qubits)
        19. phase_damping(self, lamda, qubits)
        20. i(self, qubits)
        21. x(self, qubits)
        22. y(self, qubits)
        23. z(self, qubits)
        24. h(self, qubits)
        25. s(self, qubits)
        26. t(self, qubits)
        27. sdg(self, qubits)
        28. tdg(self, qubits)
        29. rx(self, theta, qubits)
        30. ry(self, theta, qubits)
        31. rz(self, theta, qubits)
        32. p(self, phi, qubits)
        33. u1(self, lamda, qubits)
        34. u2(self, phi, lamda, qubits)
        35. u3(self, theta, phi, lamda, qubits)
        36. u(self, theta, phi, lamda, gamma, qubits)
        37. swap(self, qubit1, qubit2)
        38. cx(self, qubits1, qubits2)
        39. cy(self, qubits1, qubits2)
        40. cz(self, qubits1, qubits2)
        41. ch(self, qubits1, qubits2)
        42. cs(self, qubits1, qubits2)
        43. ct(self, qubits1, qubits2)
        44. csdg(self, qubits1, qubits2)
        45. ctdg(self, qubits1, qubits2)
        46. crx(self, theta, qubits1, qubits2)
        47. cry(self, theta, qubits1, qubits2)
        48. crz(self, theta, qubits1, qubits2)
        49. cp(self, phi, qubits1, qubits2)
        50. fsim(self, theta, phi, qubits1, qubits2)
        51. cu1(self, lamda, qubits1, qubits2)
        52. cu2(self, phi, lamda, qubits1, qubits2)
        53. cu3(self, theta, phi, lamda, qubits1, qubits2)
        54. cu(self, theta, phi, lamda, gamma, qubits1, qubits2)
        55. cswap(self, qubit1, qubit2, qubit3)
        56. ccx(self, qubits1, qubits2, qubits3)
        57. ccy(self, qubits1, qubits2, qubits3)
        58. ccz(self, qubits1, qubits2, qubits3)
        59. cch(self, qubits1, qubits2, qubits3)
        60. ccs(self, qubits1, qubits2, qubits3)
        61. cct(self, qubits1, qubits2, qubits3)
        62. ccsdg(self, qubits1, qubits2, qubits3)
        63. cctdg(self, qubits1, qubits2, qubits3)
        64. ccrx(self, theta, qubits1, qubits2, qubits3)
        65. ccry(self, theta, qubits1, qubits2, qubits3)
        66. ccrz(self, theta, qubits1, qubits2, qubits3)
        67. ccp(self, phi, qubits1, qubits2, qubits3)
        68. cfsim(self, theta, phi, qubits1, qubits2, qubits3)
        69. ccu1(self, lamda, qubits1, qubits2, qubits3)
        70. ccu2(self, phi, lamda, qubits1, qubits2, qubits3)
        71. ccu3(self, theta, phi, lamda, qubits1, qubits2, qubits3)
        72. ccu(self, theta, phi, lamda, gamma, qubits1, qubits2, qubits3)
    '''
    backend = 'superoperator'
                       

    def _apply_1q_(self, op, qubit):
        _Basic_qcircuit_._apply_1q_(self, op, qubit)
        global alp, ALP

        s = 'yzYZ'
        s0 = alp[0:self.num_qubits]
        s1 = ALP[0:self.num_qubits]
        s0 = s0.replace(s0[self.num_qubits-qubit-1], 'Y')
        s1 = s1.replace(s1[self.num_qubits-qubit-1], 'Z')
        start = s + ',' + s0 + s1
        s0 = s0.replace('Y', 'y')
        s1 = s1.replace('Z', 'z')
        end = s0 + s1

        if op.category != 'superop': op = to_superop(op)
        rep = op.represent.reshape([2]*4*op.num_qubits)
        self.state = self.state.reshape([2]*2*self.num_qubits)
        self.state = np.einsum(start+'->'+end, rep, self.state).reshape(
            [2**self.num_qubits]*2)
        return None


    def _apply_2q_(self, op, qubit1, qubit2):
        _Basic_qcircuit_._apply_2q_(self, op, qubit1, qubit2)
        global alp, ALP

        s = 'wxyzWXYZ'
        s0 = alp[0:self.num_qubits]
        s1 = ALP[0:self.num_qubits]
        s0 = s0.replace(s0[self.num_qubits-qubit1-1], 'W')
        s0 = s0.replace(s0[self.num_qubits-qubit2-1], 'X')
        s1 = s1.replace(s1[self.num_qubits-qubit1-1], 'Y')
        s1 = s1.replace(s1[self.num_qubits-qubit2-1], 'Z')
        start = s + ',' + s0 + s1
        s0 = s0.replace('X', 'x').replace('W', 'w')
        s1 = s1.replace('Z', 'z').replace('Y', 'y')
        end = s0 + s1

        if op.category != 'superop': op = to_superop(op)
        rep = op.represent.reshape([2]*4*op.num_qubits)
        self.state = self.state.reshape([2]*2*self.num_qubits)
        self.state = np.einsum(start+'->'+end, rep, self.state).reshape(
            [2**self.num_qubits]*2)
        return None


    def _apply_3q_(self, op, qubit1, qubit2, qubit3):
        _Basic_qcircuit_._apply_3q_(self, op, qubit1, qubit2, qubit3)
        global alp, ALP

        s = 'uvwxyzUVWXYZ'
        s0 = alp[0:self.num_qubits]
        s1 = ALP[0:self.num_qubits]
        s0 = s0.replace(s0[self.num_qubits-qubit1-1], 'U')
        s0 = s0.replace(s0[self.num_qubits-qubit2-1], 'V')
        s0 = s0.replace(s0[self.num_qubits-qubit3-1], 'W')
        s1 = s1.replace(s1[self.num_qubits-qubit1-1], 'X')
        s1 = s1.replace(s1[self.num_qubits-qubit2-1], 'Y')
        s1 = s1.replace(s1[self.num_qubits-qubit3-1], 'Z')
        start = s + ',' + s0 + s1
        s0 = s0.replace('U', 'u').replace('V', 'v').replace('W', 'w')
        s1 = s1.replace('X', 'x').replace('Y', 'y').replace('Z', 'z')
        end = s0 + s1

        if op.category != 'superop': op = to_superop(op)
        rep = op.represent.reshape([2]*4*op.num_qubits)
        self.state = self.state.reshape([2]*2*self.num_qubits)
        self.state = np.einsum(start+'->'+end, rep, self.state).reshape(
            [2**self.num_qubits]*2)
        return None

