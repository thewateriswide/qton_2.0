# 
# This code is part of Qton.
# Qton Version: 2.0.0
# 
# File:   _basic_qcircuit_.py
# Author: Yunheng Ma
# Date :  2022-01-27
#


import numpy as np


__all__ = ["_Basic_qcircuit_"]


from qton.operators.gates import *


class _Basic_qcircuit_:
    '''
    Basic of quantum circuits.
    
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
    backend = ''
    num_qubits = 0
    state = None


    def __init__(self, num_qubits=1):
        '''
        Initialization.
        
        -In(1):
            1. num_qubits --- number of qubits.
                type: int
                
        -Influenced(2):
            1. self.num_qubits --- number of qubits.
                type: int
            2. self.state --- circuit state representation.
                type: numpy.ndarray, complex
        '''
        self.num_qubits = num_qubits
        return None


    def _apply_1q_(self, op, qubit):
        '''
        Apply a single-qubit operation on a given qubit.

        -In(2):
            1. op --- single-qubit operation.
                type: qton operator
            2. qubit --- qubit index.
                type: int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex
        '''
        if qubit >= self.num_qubits:
            raise Exception('Qubit index oversteps.')
        return None


    def _apply_2q_(self, op, qubit1, qubit2):
        '''
        Apply a double-qubit operation on two given qubits.

        -In(3):
            1. op --- double-qubit operation.
                type: qton operator
            2. qubit1 --- first qubit index.
                type: int
            3. qubit2 --- second qubit index.
                type: int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex
        '''
        if qubit1 >= self.num_qubits or qubit2 >= self.num_qubits:
            raise Exception('Qubit index oversteps.')

        if qubit1 == qubit2:
            raise Exception('Cannot be same qubits.')
        return None


    def _apply_3q_(self, op, qubit1, qubit2, qubit3):
        '''
        Apply a triple-qubit operation on three given qubits.

        -In(4):
            1. op --- triple-qubit operation.
                type: qton operator
            2. qubit1 --- first qubit index.
                type: int
            3. qubit2 --- second qubit index.
                type: int
            4. qubit3 --- third qubit index.
                type: int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex
        '''
        if len({qubit1, qubit2, qubit3}) < 3:
            raise Exception('Cannot be same qubits.')

        if qubit1 >= self.num_qubits or \
           qubit2 >= self.num_qubits or \
           qubit3 >= self.num_qubits:
            raise Exception('Qubit index oversteps.')
        return None


    def apply_1q(self, op, qubits):
        '''
        Apply a single-qubit operation on given qubits.

        -In(2):
            1. op --- single-qubit operation.
                type: qton operator
            2. qubits --- qubit indices.
                type: int; list, int
                
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        if type(qubits) is int:
            q = [qubits]
        else:
            q = list(qubits)

        for i in q:
            self._apply_1q_(op, i)
        return None


    def apply_2q(self, op, qubits1, qubits2):
        '''
        Apply a double-qubit operation on given qubits.

        Can be one controls many or many controls one.

        -In(3):
            1. op --- double-qubit operation.
                type: qton operator
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        if type(qubits1) is int:
            q1 = [qubits1]
        else:
            q1 = list(qubits1)

        if type(qubits2) is int:
            q2 = [qubits2]
        else:
            q2 = list(qubits2)

        if len(q1) > 1 and len(q2) > 1:
            raise Exception('Too many controls or tagerts.')

        for i in q1:
            for j in q2:
                self._apply_2q_(op, i, j)
        return None


    def apply_3q(self, op, qubits1, qubits2, qubits3):
        '''
        Apply a triple-qubit operation on given qubits.

        "qubits1", "qubits2" and "qubits3", only one of them can be plural.

        -In(4):
            1. op --- triple-qubit operation.
                type: qton operator
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int
            4. qubits3 --- third qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        if type(qubits1) is int:
            q1 = [qubits1]
        else:
            q1 = list(qubits1)

        if type(qubits2) is int:
            q2 = [qubits2]
        else:
            q2 = list(qubits2)

        if type(qubits3) is int:
            q3 = [qubits3]
        else:
            q3 = list(qubits3)

        if [len(q1), len(q2), len(q3)].count(1) < 2:
            raise Exception('Too many controls or tagerts.')

        for i in q1:
            for j in q2:
                for k in q3:
                    self._apply_3q_(op, i, j, k)
        return None


    def apply(self, op, *qubits):
        '''
        Apply an operation on given qubits.
        
        -In(2):
            1. op --- qubit operation.
                type: qton operator
            2. qubits --- qubit indices.
                type: int, list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex   
        '''
        if op.num_qubits == 1:
            self.apply_1q(op, *qubits)
        elif op.num_qubits == 2:
            self.apply_2q(op, *qubits)
        elif op.num_qubits == 3:
            self.apply_3q(op, *qubits)
        else:
            raise Exception('Unsupported operation.')
        return None


    def measure(self, qubit, delete=False):
        '''
        Projective measurement on a given qubit.

        For clarity, only allow one qubit to be measured.
        
        -In(2):
            1. qubit --- index of measured qubit.
                type: int
            2. delete --- delete this qubit after measurement?
                type: bool

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex
                
        -Return(1):
            1. bit --- measured output, 0 or 1.
                type: int
        '''
        if delete and self.num_qubits < 2:
            raise Exception('Cannot delete the last qubit.')
        return None


    def add_qubit(self, num_qubits=1):
        '''
        Add qubit(s) at the tail of the circuit.
        
        -In(1):
            1. num_qubits --- number of qubits to be added in.
                type: int

        -Influenced(2):
            1. self.num_qubits --- number of qubits.
                type: int
            2. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        if self.backend == 'statevector':
            new = np.zeros(2**num_qubits, complex)
            new[0] = 1.
        elif self.backend == 'density_matrix':
            new = np.zeros((2**num_qubits, 2**num_qubits), complex)
            new[0, 0] = 1.
        elif self.backend == 'unitary':
            new = np.eye((2**num_qubits), dtype=complex)
        elif self.backend == 'superoperator':
            new = np.zeros((2**num_qubits, 2**num_qubits), complex)
            new[0, 0] = 1.
        else:
            raise Exception('Unrecognized circuit backend.')
            
        self.state = np.kron(new, self.state)
        self.num_qubits += num_qubits
        return None


    def sample(self, shots=1024, output='memory'):
        '''
        Sample a statevector, sampling with replacement.
        
        -In(2):
            1. shots --- sampling times.
                type: int
            2. output --- output date type
                type: str: "memory", "statistic", "counts"

        -Return(1):
            1.(3):
                1. memory --- every output.
                    type: list, int
                2. statistic --- repeated times of every basis.
                    type: numpy.ndarray, int
                3. counts --- counts for basis.
                    type: dict            
        '''
        if self.backend == 'statevector':
            distribution = self.state * self.state.conj()
        elif self.backend == 'density_matrix':
            distribution = self.state.diagonal()
        elif self.backend == 'unitary':
            distribution = self.state[:, 0] * self.state[:, 0].conj()
        elif self.backend == 'superoperator':
            distribution = self.state.diagonal()
        else:
            raise Exception('Unrecognized circuit backend.')

        from random import choices
        N = 2**self.num_qubits
        memory = choices(range(N), weights=distribution, k=shots)

        if output == 'memory':
            return memory

        elif output == 'statistic':
            statistic = np.zeros(N, int)
            for i in memory:
                statistic[i] += 1
            return statistic

        elif output == 'counts':
            counts = {}
            for i in memory:
                key = format(i, '0%db' % self.num_qubits)
                if key in counts:
                    counts[key] += 1
                else:
                    counts[key] = 1
            return counts

        else:
            raise Exception('Unrecognized output type.')


# 
# Gate methods.
# 

    def i(self, qubits):
        '''
        Identity gate.
        
        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(I_gate(), qubits)
        return None


    def x(self, qubits):
        '''
        Pauli-X gate.
        
        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(X_gate(), qubits)
        return None


    def y(self, qubits):
        '''
        Pauli-Y gate.
        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(Y_gate(), qubits)
        return None


    def z(self, qubits):
        '''
        Pauli-Z gate.

        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(Z_gate(), qubits)
        return None


    def h(self, qubits):
        '''
        Hadamard gate.

        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(H_gate(), qubits)
        return None


    def s(self, qubits):
        '''
        Phase S gate.

        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(S_gate(), qubits)
        return None


    def t(self, qubits):
        '''
        pi/8 T gate.

        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(T_gate(), qubits)
        return None


    def sdg(self, qubits):
        '''
        S dagger gate.
        
        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(S_gate(dagger=True), qubits)
        return None


    def tdg(self, qubits):
        '''
        T dagger gate.
        
        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(T_gate(dagger=True), qubits)
        return None


    def rx(self, theta, qubits):
        '''
        Rotation along X axis.
        
        -In(2):
            1. theta --- rotation angle.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(Rx_gate([theta]), qubits)
        return None


    def ry(self, theta, qubits):
        '''
        Rotation along Y axis.
        
        -In(2):
            1. theta --- rotation angle.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(Ry_gate([theta]), qubits)
        return None


    def rz(self, theta, qubits):
        '''
        Rotation along Z axis.
        
        -In(2):
            1. theta --- rotation angle.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(Rz_gate([theta]), qubits)
        return None


    def p(self, phi, qubits):
        '''
        Phase gate.

        -In(2):
            1. phi --- phase angle.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(P_gate([phi]), qubits)
        return None


    def u1(self, lamda, qubits):
        '''
        U1 gate.
        
        -In(2):
            1. lamda --- phase angle.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(U1_gate([lamda]), qubits)
        return None


    def u2(self, phi, lamda, qubits):
        '''
        U2 gate.
        
        -In(3):
            1. phi --- phase angle.
                type: float
            2. lamda --- phase angle.
                type: float
            3. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(U2_gate([phi, lamda]), qubits)
        return None


    def u3(self, theta, phi, lamda, qubits):
        '''
        U3 gate.

        -In(4):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(U3_gate([theta, phi, lamda]), qubits)
        return None


    def u(self, theta, phi, lamda, gamma, qubits):
        '''
        Universal gate.
        
        -In(5):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. gamma --- global phase.
                type: float
            5. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_1q(U_gate([theta, phi, lamda, gamma]), qubits)
        return None


    def swap(self, qubit1, qubit2):
        '''
        Swap gate.
        
        -In(2):
            1. qubit1 --- first qubit index.
                type: int
            2. qubit2 --- second qubit index.
                type: int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self._apply_2q_(Swap_gate(), qubit1, qubit2)
        return None


    def cx(self, qubits1, qubits2):
        '''
        Controlled Pauli-X gate.
        
        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(X_gate(num_ctrls=1), qubits1, qubits2)
        return None


    def cy(self, qubits1, qubits2):
        '''
        Controlled Pauli-Y gate.
        
        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(Y_gate(num_ctrls=1), qubits1, qubits2)
        return None


    def cz(self, qubits1, qubits2):
        '''
        Controlled Pauli-Z gate.
        
        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(Z_gate(num_ctrls=1), qubits1, qubits2)
        return None


    def ch(self, qubits1, qubits2):
        '''
        Controlled Hadamard gate.
        
        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(H_gate(num_ctrls=1), qubits1, qubits2)
        return None


    def cs(self, qubits1, qubits2):
        '''
        Controlled S gate.
        
        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(S_gate(num_ctrls=1), qubits1, qubits2)
        return None


    def ct(self, qubits1, qubits2):
        '''
        Controlled T gate.
        
        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(T_gate(num_ctrls=1), qubits1, qubits2)
        return None


    def csdg(self, qubits1, qubits2):
        '''
        Controlled S dagger gate.

        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(S_gate(num_ctrls=1, dagger=True), qubits1, qubits2)
        return None


    def ctdg(self, qubits1, qubits2):
        '''
        Controlled T dagger gate.
        
        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(T_gate(num_ctrls=1, dagger=True), qubits1, qubits2)
        return None


    def crx(self, theta, qubits1, qubits2):
        '''
        Controlled rotation along X axis.
        
        -In(3):
            1. theta --- rotation angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(Rx_gate([theta], num_ctrls=1), qubits1, qubits2)
        return None


    def cry(self, theta, qubits1, qubits2):
        '''
        Controlled rotation along Y axis.
        
        -In(3):
            1. theta --- rotation angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(Ry_gate([theta], num_ctrls=1), qubits1, qubits2)
        return None


    def crz(self, theta, qubits1, qubits2):
        '''
        Controlled rotation along Z axis.
        
        -In(3):
            1. theta --- rotation angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(Rz_gate([theta], num_ctrls=1), qubits1, qubits2)
        return None


    def cp(self, phi, qubits1, qubits2):
        '''
        Controlled phase gate.
        
        -In(3):
            1. phi --- phase angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(P_gate([phi], num_ctrls=1), qubits1, qubits2)
        return None


    def fsim(self, theta, phi, qubits1, qubits2):
        '''
        fSim gate.
        
        -In(3):
            1. theta --- rotation angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. qubits1 --- first qubit indices.
                type: int; list, int
            4. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(Fsim_gate([theta, phi]), qubits1, qubits2)
        return None


    def cu1(self, lamda, qubits1, qubits2):
        '''
        Controlled U1 gate.

        -In(3):
            1. lamda --- phase angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(U1_gate([lamda], num_ctrls=1), qubits1, qubits2)
        return None


    def cu2(self, phi, lamda, qubits1, qubits2):
        '''
        Controlled U2 gate.
        
        -In(4):
            1. phi --- phase angle.
                type: float
            2. lamda --- phase angle.
                type: float
            3. qubits1 --- first qubit indices.
                type: int; list, int
            4. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(U2_gate([phi, lamda], num_ctrls=1), qubits1, qubits2)
        return None


    def cu3(self, theta, phi, lamda, qubits1, qubits2):
        '''
        Controlled U3 gate.
        
        -In(5):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. qubits1 --- first qubit indices.
                type: int; list, int
            5. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(U3_gate([theta, phi, lamda], num_ctrls=1), qubits1, qubits2)
        return None


    def cu(self, theta, phi, lamda, gamma, qubits1, qubits2):
        '''
        Controlled universal gate.

        -In(6):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. gamma --- global phase.
                type: float
            5. qubits1 --- first qubit indices.
                type: int; list, int
            6. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_2q(U_gate([theta, phi, lamda, gamma], num_ctrls=1), qubits1, qubits2)
        return None


    def cswap(self, qubit1, qubit2, qubit3):
        '''
        Controlled swap gate.
        
        -In(3):
            1. qubit1 --- first qubit index.
                type: int
            2. qubit2 --- second qubit index.
                type: int
            3. qubit3 --- third qubit index.
                type: int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self._apply_3q_(Swap_gate(num_ctrls=1), qubit1, qubit2, qubit3)
        return None


    def ccx(self, qubits1, qubits2, qubits3):
        '''
        Double controlled Pauli-X gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(X_gate(num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccy(self, qubits1, qubits2, qubits3):
        '''
        Double controlled Pauli-Y gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(Y_gate(num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccz(self, qubits1, qubits2, qubits3):
        '''
        Double controlled Pauli-Z gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(Z_gate(num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def cch(self, qubits1, qubits2, qubits3):
        '''
        Double controlled Hadamard gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(H_gate(num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccs(self, qubits1, qubits2, qubits3):
        '''
        Double controlled S gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(S_gate(num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def cct(self, qubits1, qubits2, qubits3):
        '''
        Double controlled T gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(T_gate(num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccsdg(self, qubits1, qubits2, qubits3):
        '''
        Double controlled S dagger gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(S_gate(num_ctrls=2, dagger=True), qubits1, qubits2, qubits3)
        return None


    def cctdg(self, qubits1, qubits2, qubits3):
        '''
        Double controlled T dagger gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(T_gate(num_ctrls=2, dagger=True), qubits1, qubits2, qubits3)
        return None


    def ccrx(self, theta, qubits1, qubits2, qubits3):
        '''
        Double controlled rotation along X axis.
        
        -In(4):
            1. theta --- rotation angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int
            4. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(Rx_gate([theta], num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccry(self, theta, qubits1, qubits2, qubits3):
        '''
        Double controlled rotation along Y axis.
        
        -In(4):
            1. theta --- rotation angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int
            4. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(Ry_gate([theta], num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccrz(self, theta, qubits1, qubits2, qubits3):
        '''
        Double controlled rotation along Z axis.
        
        -In(4):
            1. theta --- rotation angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int
            4. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(Rz_gate([theta], num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccp(self, phi, qubits1, qubits2, qubits3):
        '''
        Double controlled phase gate.
        
        -In(4):
            1. phi --- phase angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int
            4. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(P_gate([phi], num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def cfsim(self, theta, phi, qubits1, qubits2, qubits3):
        '''
        Controlled fSim gate.
        
        -In(5):
            1. theta --- rotation angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. qubits1 --- first qubit indices.
                type: int; list, int
            4. qubits2 --- second qubit indices.
                type: int; list, int
            5. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(Fsim_gate([theta, phi], num_ctrls=1), qubits1, qubits2, qubits3)
        return None


    def ccu1(self, lamda, qubits1, qubits2, qubits3):
        '''
        Double controlled U1 gate.
        
        -In(4):
            1. lamda --- phase angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int
            4. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(U1_gate([lamda], num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccu2(self, phi, lamda, qubits1, qubits2, qubits3):
        '''
        Double controlled U2 gate.
        
        -In(5):
            1. phi --- phase angle.
                type: float
            2. lamda --- phase angle.
                type: float
            3. qubits1 --- first qubit indices.
                type: int; list, int
            4. qubits2 --- second qubit indices.
                type: int; list, int
            5. qubits3 --- third qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(U2_gate([phi, lamda], num_ctrls=2), qubits1, qubits2,qubits3)
        return None


    def ccu3(self, theta, phi, lamda, qubits1, qubits2, qubits3):
        '''
        Double controlled U3 gate.
        
        -In(6):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. qubits1 --- first qubit indices.
                type: int; list, int
            5. qubits2 --- second qubit indices.
                type: int; list, int
            6. qubits3 --- third qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(U3_gate([theta, phi, lamda], num_ctrls=2), qubits1, qubits2, 
            qubits3)
        return None


    def ccu(self, theta, phi, lamda, gamma, qubits1, qubits2, qubits3):
        '''
        Double controlled universal gate.
        
        -In(7):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. gamma --- global phase.
                type: float
            5. qubits1 --- first qubit indices.
                type: int; list, int
            6. qubits2 --- second qubit indices.
                type: int; list, int
            7. qubits3 --- third qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply_3q(U_gate([theta, phi, lamda, gamma], num_ctrls=2), qubits1, 
            qubits2, qubits3)
        return None
