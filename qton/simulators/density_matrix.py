# 
# This code is part of Qton.
# Qton Version: 2.0.0
# 
# File:   density_matrix.py
# Author: Yunheng Ma
# Date :  2022-01-27
#


import numpy as np


__all__ = ["svec2dmat",
           "random_qubit",
           "expectation",
           "operate",
           "Qdensity_matrix",
           ]


def svec2dmat(svec):
    '''
    Statevector to density matrix.
    
    $$
    \rho = |\psi\rangle \langle\psi|
    $$
    
    -In(1):
        1. svec --- quantum statevector.
            type: numpy.ndarray, 1D, complex
    
    -Return(1):
        1. --- density matrix corresponding to the statevector.
            type: numpy.ndarray, 2D, complex
    '''
    # return np.enisum('i,j->ij', svec, svec.conj())
    return np.outer(svec, svec.conj())


def random_qubit(mixed=True):
    '''
    Returns a random single-qubit density matrix. 
    
    $$
    \rho = \frac{1}{2}(I + r_x \sigma_x + r_y \sigma_y + r_z \sigma_z)
    $$

    $r_x^2 + r_y^2 + r_z^2 < 1$ for mixed state.

    -In(1):
        1. mixed --- a mixed state?
            type: bool

    -Return(1):
        1. dmat --- single-qubit density matrix.
            type: numpy.ndarray, 2D, complex
    '''
    x, y, z = np.random.standard_normal(3)
    r = np.sqrt(x**2 + y**2 + z**2)
    if r == 0:
        rx = ry = rz = 0.
    else:
        rx, ry, rz = x/r, y/r, z/r

    if mixed:
        e = np.random.random()
        rx, ry, rz = rx*e, ry*e, rz*e

    dmat = np.zeros((2,2), complex)
    dmat[0, 0] = 1. + rz
    dmat[0, 1] = rx - 1j*ry
    dmat[1, 0] = rx + 1j*ry
    dmat[1, 1] = 1. - rz 
    dmat *= 0.5
    return dmat
    

def expectation(oper, dmat):
    '''
    Expectation of quantum operations on a density matrix.

    $$
    \langle E \rangle = \sum_k {\rm Tr}(E_k\rho)
    $$
    
    -In(2):
        1. oper --- quantum operations.
            type: list, numpy.ndarray, 2D, complex
        2. dmat --- density matrix of system.
            type: numpy.ndarray, 2D, complex

    -Return(1):
        1. ans --- expectation value.
            type: complex
    '''
    if type(oper) is not list:
        oper = [oper]

    ans = 0.
    for i in range(len(oper)):
        # ans += np.einsum('ij,ji->', oper[i], dmat)
        ans += np.trace(np.matmul(oper[i], dmat))
    return ans


def operate(oper, dmat):
    '''
    Implement a single-qubit or double-qubit quantum operation.

    $$
    \rho' = \sum_k E_k \rho E_k^\dagger
    $$

    -In(2):
        1. oper --- the quantum operations.
            type: list, numpy.ndarray, 2D, complex
        2. dmat --- density matrix.
            type: numpy.ndarray, 2D, complex
            
    -Return(1):
        1. ans --- density matrix after implementation.
            type: numpy.ndarray, 2D, complex
    '''
    if type(oper) is not list:
        oper = [oper]

    n = len(oper)
    ans = np.zeros(dmat.shape, complex)
    for i in range(n):
        # ans += np.einsum('ij,jk,lk->il', oper[i], dmat, oper[i].conj())
        ans += np.matmul(oper[i],
            np.matmul(dmat, oper[i].transpose().conj()))
    return ans


# alphabet
alp = 'abcdefghijklmnopqrstuvwxyz'
ALP = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


from ._basic_qcircuit_ import _Basic_qcircuit_
from qton.operators.channels import *
from qton.operators.channels import to_channel


class Qdensity_matrix(_Basic_qcircuit_):
    '''
    Quantum circuit represented by circuit density matrix.

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
    backend = 'density_matrix'
         

    def __init__(self, num_qubits=1):
        super().__init__(num_qubits)
        self.state = np.zeros((2**num_qubits, 2**num_qubits), complex)
        self.state[0, 0] = 1.0
        return None


    def _apply_1q_(self, op, qubit):
        super()._apply_1q_(op, qubit)
        global alp, ALP

        sa = 'yY'
        sb = 'Zz'
        s0 = alp[0:self.num_qubits]
        s1 = ALP[0:self.num_qubits]
        s0 = s0.replace(s0[self.num_qubits-qubit-1], 'Y')
        s1 = s1.replace(s1[self.num_qubits-qubit-1], 'Z')
        start = sa + ',' + s0 + s1 + ',' + sb
        s0 = s0.replace('Y', 'y')
        s1 = s1.replace('Z', 'z')
        end = s0 + s1

        if op.category != 'channel': op = to_channel(op)
        self.state = self.state.reshape([2]*2*self.num_qubits)
        state = np.zeros(self.state.shape, complex)
        for i in range(len(op.represent)):
            rep_a = op.represent[i].reshape([2]*2*op.num_qubits)
            rep_b = op.represent[i].transpose().conj().reshape(
                [2]*2*op.num_qubits)
            state += np.einsum(start+'->'+end, rep_a, self.state, rep_b)
        self.state = state.reshape(2**self.num_qubits, -1)
        return None


    def _apply_2q_(self, op, qubit1, qubit2):
        super()._apply_2q_(op, qubit1, qubit2)

        sa = 'wxWX'
        sb = 'YZyz'
        s0 = alp[0:self.num_qubits]
        s1 = ALP[0:self.num_qubits]
        s0 = s0.replace(s0[self.num_qubits-qubit1-1], 'W')
        s1 = s1.replace(s1[self.num_qubits-qubit1-1], 'Y')
        s0 = s0.replace(s0[self.num_qubits-qubit2-1], 'X')
        s1 = s1.replace(s1[self.num_qubits-qubit2-1], 'Z')
        start = sa + ',' + s0  + s1 + ',' + sb
        s0 = s0.replace('X', 'x').replace('W', 'w')
        s1 = s1.replace('Y', 'y').replace('Z', 'z')
        end = s0 + s1

        if op.category != 'channel': op = to_channel(op)
        self.state = self.state.reshape([2]*2*self.num_qubits)
        state = np.zeros(self.state.shape, complex)
        for i in range(len(op.represent)):
            rep_a = op.represent[i].reshape([2]*2*op.num_qubits)
            rep_b = op.represent[i].transpose().conj().reshape(
                [2]*2*op.num_qubits)
            state += np.einsum(start+'->'+end, rep_a, self.state, rep_b)
        self.state = state.reshape(2**self.num_qubits, -1)
        return None


    def _apply_3q_(self, op, qubit1, qubit2, qubit3):
        super()._apply_3q_(op, qubit1, qubit2, qubit3)

        sa = 'uvwUVW'
        sb = 'XYZxyz'
        s0 = alp[0:self.num_qubits]
        s1 = ALP[0:self.num_qubits]
        s0 = s0.replace(s0[self.num_qubits-qubit1-1], 'U')
        s1 = s1.replace(s1[self.num_qubits-qubit1-1], 'X')
        s0 = s0.replace(s0[self.num_qubits-qubit2-1], 'V')
        s1 = s1.replace(s1[self.num_qubits-qubit2-1], 'Y')
        s0 = s0.replace(s0[self.num_qubits-qubit3-1], 'W')
        s1 = s1.replace(s1[self.num_qubits-qubit3-1], 'Z')
        start = sa + ',' + s0  + s1 + ',' + sb
        s0 = s0.replace('W', 'w').replace('V', 'v').replace('U', 'u')
        s1 = s1.replace('X', 'x').replace('Y', 'y').replace('Z', 'z')
        end = s0 + s1

        if op.category != 'channel': op = to_channel(op)
        self.state = self.state.reshape([2]*2*self.num_qubits)
        state = np.zeros(self.state.shape, complex)
        for i in range(len(op.represent)):
            rep_a = op.represent[i].reshape([2]*2*op.num_qubits)
            rep_b = op.represent[i].transpose().conj().reshape(
                [2]*2*op.num_qubits)
            state += np.einsum(start+'->'+end, rep_a, self.state, rep_b)
        self.state = state.reshape(2**self.num_qubits, -1)
        return None


    def measure(self, qubit, delete=False):
        super().measure(qubit, delete)
        state = self.state.reshape([2]*2*self.num_qubits)
        dic = locals()

        string00 = ':, '*(self.num_qubits-qubit-1) + '0, ' + \
            ':, '*(self.num_qubits-1) + '0, ' + ':, '*qubit
        string11 = ':, '*(self.num_qubits-qubit-1) + '1, ' + \
            ':, '*(self.num_qubits-1) + '1, ' + ':, '*qubit

        string01 = ':, '*(self.num_qubits-qubit-1) + '0, ' + \
            ':, '*(self.num_qubits-1) + '1, ' + ':, '*qubit
        string10 = ':, '*(self.num_qubits-qubit-1) + '1, ' + \
            ':, '*(self.num_qubits-1) + '0, ' + ':, '*qubit

        exec('reduced0 = state[' + string0 + ']', dic)
        measured = dic['reduced0'].reshape(2**(self.num_qubits-1), -1)
        probability0 = np.trace(measured)

        if np.random.random() < probability0:
            bit = 0
            if delete:
                self.state = measured
                self.num_qubits -= 1
            else:
                exec('state[' + string01 + '] = 0.', dic)
                exec('state[' + string10 + '] = 0.', dic)
                exec('state[' + string11 + '] = 0.', dic)
                self.state = dic['state'].reshape(2**self.num_qubits, -1)
            self.state /= probability0
        else:
            bit = 1
            if delete:
                exec('reduced1 = state[' + string1 + ']', dic)
                self.state = dic['reduced1'].reshape(2**(self.num_qubits-1),-1)
                self.num_qubits -= 1
            else:
                exec('state[' + string00 + '] = 0.', dic)
                exec('state[' + string01 + '] = 0.', dic)
                exec('state[' + string10 + '] = 0.', dic)
                self.state = dic['state'].reshape(2**self.num_qubits, -1)
            self.state /= (1. - probability0)
        return bit


    def reduce(self, qubit):
        '''
        Recuced density matrix after a partial trace over a given qubit.
        
        -In(1):
            1. qubit --- qubit index.
                type: int
                
        -Influenced(2):
            1. self.state --- qubit density matrix.
                type: numpy.ndarray, 2D, complex
            2. self.num_qubit --- number of qubits.
                type: int
        '''
        global alp, ALP
        s0 = alp[0:self.num_qubits]
        s1 = ALP[0:self.num_qubits]

        s1=s1.replace(s1[self.num_qubits-qubit-1], s0[self.num_qubits-qubit-1])
        start = s0 + s1

        s0 = s0.replace(s0[self.num_qubits-qubit-1], '')
        s1 = s1.replace(s1[self.num_qubits-qubit-1], '')
        end = s0 + s1

        self.state = self.state.reshape([2]*2*self.num_qubits)
        self.state = np.einsum(
            start+'->'+end, self.state).reshape([2**(self.num_qubits-1)]*2)
        self.num_qubits -= 1
        return None


# 
# Kraus channel methods.
# 

    def bit_flip(self, p, qubits):
        '''
        Bit flip channel.

        -In(2):
            1. p --- the probability for not flipping.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
                
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(Bit_flip([p]), qubits)
        return None


    def phase_flip(self, p, qubits):
        '''
        Phase flip channel.
        
        -In(2):
            1. p --- the probability for not flipping.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(Phase_flip([p]), qubits)
        return None


    def bit_phase_flip(self, p, qubits):
        '''
        Bit phase flip channel.
        
        -In(2):
            1. p --- the probability for not flipping.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(Bit_phase_flip([p]), qubits)
        return None


    def depolarize(self, p, qubits):
        '''
        Depolarizing channel.
    
        -In(2):
            1. p --- the probability for depolarization.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(Depolarize([p]), qubits)
        return None


    def amplitude_damping(self, gamma, qubits):
        '''
        Amplitude damping channel.
        
        -In(2):
            1. gamma --- probability such as losing a photon.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(Amplitude_damping([gamma]), qubits)
        return None


    def generalized_amplitude_damping(self, p, gamma, qubits):
        '''
        Generalized amplitude damping channel.
        
        -In(3):
            1. p --- the probability for acting normal amplitude damping.
                type: float
            2. gamma --- probability such as losing a photon.
                type: float
            3. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(Generalized_amplitude_damping([p, gamma]), qubits)
        return None


    def phase_damping(self, lamda, qubits):
        '''
        Phase damping channel.
        
        -In(2):
            1. lamda --- probability such as a photon from the system has been 
                scattered(without loss of energy).
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(Phase_damping([lamda]), qubits)
        return None
