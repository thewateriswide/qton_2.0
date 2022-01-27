# 
# This code is part of Qton.
# Qton Version: 2.0.0
# 
# File:   __init__.py
# Author: Yunheng Ma
# Date :  2022-01-27
#


__version__ = '2.0.0'
__author__ = 'Yunheng Ma'


__all__ = ["Qcircuit"]


from .simulators.statevector import Qstatevector
from .simulators.unitary import Qunitary
from .simulators.density_matrix import Qdensity_matrix
from .simulators.superoperator import Qsuperoperator


def Qcircuit(num_qubits, backend='statevector'):
    '''
    Create new quantum circuit instance.
    
    -In(2):
        1. num_qubits --- number of qubits.
            type: int
        2. backend --- how to execute the circuit; 'statevector', 
            'density_matrix', 'unitary', or 'superoperator'.
            type: str
    
    -Return(1):
        1. --- quantum circuit instance.
            type: qton circuit instance.
    '''
    if backend == 'statevector':
        return Qstatevector(num_qubits)
    if backend == 'unitary':
        return Qunitary(num_qubits)
    elif backend == 'density_matrix':
        return Qdensity_matrix(num_qubits)
    elif backend == 'superoperator':
        return Qsuperoperator(num_qubits)
    else:
        raise Exception('Unrecognized backend.')