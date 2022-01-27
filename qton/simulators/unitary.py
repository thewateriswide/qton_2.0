# 
# This code is part of Qton.
# Qton Version: 2.0.0
# 
# File:   unitary.py
# Author: Yunheng Ma
# Date :  2022-01-27
#


import numpy as np


__all__ = ["Qunitary"]


# alphabet
alp = 'abcdefghijklmnopqrstuvwxyz'
ALP = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


from ._basic_qcircuit_ import _Basic_qcircuit_


class Qunitary(_Basic_qcircuit_):
    '''
    Quantum circuit represented by circuit unitary.

    -Attributes(3):
        1. backend --- how to execute the circuit; 'statevector', 
            'density_matrix', 'unitary', or 'superoperator'.
            type: str
        2. num_qubits --- number of qubits.
            type: int
        3. state --- circuit state representation.
            type: numpy.ndarray, complex        
    
    -Methods(64):
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
        12. i(self, qubits)
        13. x(self, qubits)
        14. y(self, qubits)
        15. z(self, qubits)
        16. h(self, qubits)
        17. s(self, qubits)
        18. t(self, qubits)
        19. sdg(self, qubits)
        20. tdg(self, qubits)
        21. rx(self, theta, qubits)
        22. ry(self, theta, qubits)
        23. rz(self, theta, qubits)
        24. p(self, phi, qubits)
        25. u1(self, lamda, qubits)
        26. u2(self, phi, lamda, qubits)
        27. u3(self, theta, phi, lamda, qubits)
        28. u(self, theta, phi, lamda, gamma, qubits)
        29. swap(self, qubit1, qubit2)
        30. cx(self, qubits1, qubits2)
        31. cy(self, qubits1, qubits2)
        32. cz(self, qubits1, qubits2)
        33. ch(self, qubits1, qubits2)
        34. cs(self, qubits1, qubits2)
        35. ct(self, qubits1, qubits2)
        36. csdg(self, qubits1, qubits2)
        37. ctdg(self, qubits1, qubits2)
        38. crx(self, theta, qubits1, qubits2)
        39. cry(self, theta, qubits1, qubits2)
        40. crz(self, theta, qubits1, qubits2)
        41. cp(self, phi, qubits1, qubits2)
        42. fsim(self, theta, phi, qubits1, qubits2)
        43. cu1(self, lamda, qubits1, qubits2)
        44. cu2(self, phi, lamda, qubits1, qubits2)
        45. cu3(self, theta, phi, lamda, qubits1, qubits2)
        46. cu(self, theta, phi, lamda, gamma, qubits1, qubits2)
        47. cswap(self, qubit1, qubit2, qubit3)
        48. ccx(self, qubits1, qubits2, qubits3)
        49. ccy(self, qubits1, qubits2, qubits3)
        50. ccz(self, qubits1, qubits2, qubits3)
        51. cch(self, qubits1, qubits2, qubits3)
        52. ccs(self, qubits1, qubits2, qubits3)
        53. cct(self, qubits1, qubits2, qubits3)
        54. ccsdg(self, qubits1, qubits2, qubits3)
        55. cctdg(self, qubits1, qubits2, qubits3)
        56. ccrx(self, theta, qubits1, qubits2, qubits3)
        57. ccry(self, theta, qubits1, qubits2, qubits3)
        58. ccrz(self, theta, qubits1, qubits2, qubits3)
        59. ccp(self, phi, qubits1, qubits2, qubits3)
        60. cfsim(self, theta, phi, qubits1, qubits2, qubits3)
        61. ccu1(self, lamda, qubits1, qubits2, qubits3)
        62. ccu2(self, phi, lamda, qubits1, qubits2, qubits3)
        63. ccu3(self, theta, phi, lamda, qubits1, qubits2, qubits3)
        64. ccu(self, theta, phi, lamda, gamma, qubits1, qubits2, qubits3)
    '''
    backend = 'unitary'

    
    def __init__(self, num_qubits=1):
        super().__init__(num_qubits)
        self.state = np.eye((2**num_qubits), dtype=complex)
        return None


    def _apply_1q_(self, op, qubit):
        super()._apply_1q_(op, qubit)
        global alp, ALP

        s = 'zZ'
        s0 = alp[0:self.num_qubits]
        s1 = ALP[0:self.num_qubits]
        start = s + ',' + s0.replace(s0[self.num_qubits-qubit-1], 'Z') + s1
        end = s0.replace(s0[self.num_qubits-qubit-1], 'z') + s1

        rep = op.represent.reshape([2]*2*op.num_qubits)
        self.state = self.state.reshape([2]*2*self.num_qubits)
        self.state = np.einsum(
            start+'->'+end, rep, self.state).reshape(2**self.num_qubits, -1)
        return None


    def _apply_2q_(self, op, qubit1, qubit2):
        super()._apply_2q_(op, qubit1, qubit2)
        global alp, ALP

        s = 'yzYZ'
        s0 = alp[0:self.num_qubits]
        s1 = ALP[0:self.num_qubits]
        start = s + ',' \
            + s0.replace(s0[self.num_qubits-qubit1-1],
            'Y').replace(s0[self.num_qubits-qubit2-1], 'Z') + s1
        end = s0.replace(s0[self.num_qubits-qubit1-1],
            'y').replace(s0[self.num_qubits-qubit2-1], 'z') + s1

        rep = op.represent.reshape([2]*2*op.num_qubits)
        self.state = self.state.reshape([2]*2*self.num_qubits)
        self.state = np.einsum(
            start+'->'+end, rep, self.state).reshape(2**self.num_qubits, -1)
        return None


    def _apply_3q_(self, op, qubit1, qubit2, qubit3):
        super()._apply_3q_(op, qubit1, qubit2, qubit3)
        global alp, ALP

        s = 'xyzXYZ'
        s0 = alp[0:self.num_qubits]
        s1 = ALP[0:self.num_qubits]
        start = s + ',' \
            + s0.replace(s0[self.num_qubits-qubit1-1],
            'X').replace(s0[self.num_qubits-qubit2-1], 
            'Y').replace(s0[self.num_qubits-qubit3-1], 'Z') + s1
        end = s0.replace(s0[self.num_qubits-qubit1-1],
            'x').replace(s0[self.num_qubits-qubit2-1], 
            'y').replace(s0[self.num_qubits-qubit3-1], 'z') + s1

        rep = op.represent.reshape([2]*2*op.num_qubits)
        self.state = self.state.reshape([2]*2*self.num_qubits)
        self.state = np.einsum(
            start+'->'+end, rep, self.state).reshape(2**self.num_qubits, -1)
        return None


    def measure(self, qubit, delete=False):
        super().measure(qubit, delete)

        state = self.state.reshape([2]*2*self.num_qubits)
        svec = self.state[:, 0].reshape([2]*self.num_qubits)
        dic = locals()

        string0 = ':, '*(self.num_qubits-qubit-1) + '0, ' + ':, '*qubit
        string1 = ':, '*(self.num_qubits-qubit-1) + '1, ' + ':, '*qubit

        exec('reduced0 = svec[' + string0 + ']', dic)
        measured = dic['reduced0'].reshape(-1)
        probability0 = np.einsum('i,i->', measured, measured.conj())

        if np.random.random() < probability0:
            bit = 0
            if delete:
                exec('reduced0 = state[' + string0 + string0 + ']', dic)
                self.state = dic['reduced0'].reshape(2**(self.num_qubits-1),-1)
                self.num_qubits -= 1
            else:
                exec('state[' + string1 + ':, '*self.num_qubits + '] = 0.',dic)
                self.state = dic['state'].reshape(2**self.num_qubits, -1)
            self.state /= np.sqrt(probability0)
        else:
            bit = 1
            if delete:
                exec('reduced1 = state[' + string1 + string0 + ']', dic)
                self.state = dic['reduced1'].reshape(2**(self.num_qubits-1),-1)
                self.num_qubits -= 1
            else:
                exec('state[' + string0 + ':, '*self.num_qubits + '] = 0.',dic)
                self.state = dic['state'].reshape(2**self.num_qubits, -1)
            self.state /= np.sqrt(1. - probability0)
        return bit
