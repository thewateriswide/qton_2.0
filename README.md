# Qton

**Qton** is written in Python. 

**Qton** is for simulating and researching quantum computers.



## Version

Current version = `2.0.0`

This version has been altered a lot, compared with version `1.x.x`

This version is **NOT** beginner-friendly.  Please go back to earlier version if you need.  

A much detailed manual will come in future.



## Install

Place the `qton` folder in your working directory, then import it as 

```python
from qton import *
```

same as

```python
from qton import Qcircuit
```

`Qcircuit` is a function, it creates a quantum circuit object. You can manipulate it with gate methods.



## Requirement

**Qton** is based on _NumPy_.



## Methods

Below is an example for showing all the gate methods and usages. 

 ```python
import numpy as np
from qton import *

# number of qubits
n = 4

q = [*range(n)]
qubit1 = q.pop()
qubit2 = q.pop()
qubit3 = q.pop()

qubits = [*range(n)]

q = [*range(n)]
q1 = q.pop()
q2 = q.pop()
q3 = q
q = [q1,q2,q3]
np.random.shuffle(q)
qubits1, qubits2, qubits3 = q

theta, phi, lamda, gamma = np.random.random(4) * np.pi * 2

# create a circuit represented by a statevector
qc = Qcircuit(n, 'statevector')

qc.i(qubits)
qc.x(qubits)
qc.y(qubits)
qc.z(qubits)
qc.h(qubits)
qc.s(qubits)
qc.t(qubits)
qc.sdg(qubits)
qc.tdg(qubits)
qc.rx(theta, qubits)
qc.ry(theta, qubits)
qc.rz(theta, qubits)
qc.p(phi, qubits)
qc.u1(lamda, qubits)
qc.u2(phi, lamda, qubits)
qc.u3(theta, phi, lamda, qubits)
qc.u(theta, phi, lamda, gamma, qubits)
qc.swap(qubit1, qubit2)
qc.cx(qubits1, qubits2)
qc.cy(qubits1, qubits2)
qc.cz(qubits1, qubits2)
qc.ch(qubits1, qubits2)
qc.cs(qubits1, qubits2)
qc.ct(qubits1, qubits2)
qc.csdg(qubits1, qubits2)
qc.ctdg(qubits1, qubits2)
qc.crx(theta, qubits1, qubits2)
qc.cry(theta, qubits1, qubits2)
qc.crz(theta, qubits1, qubits2)
qc.cp(phi, qubits1, qubits2)
qc.fsim(theta, phi, qubits1, qubits2)
qc.cu1(lamda, qubits1, qubits2)
qc.cu2(phi, lamda, qubits1, qubits2)
qc.cu3(theta, phi, lamda, qubits1, qubits2)
qc.cu(theta, phi, lamda, gamma, qubits1, qubits2)
qc.cswap(qubit1, qubit2, qubit3)
qc.ccx(qubits1, qubits2, qubits3)
qc.ccy(qubits1, qubits2, qubits3)
qc.ccz(qubits1, qubits2, qubits3)
qc.cch(qubits1, qubits2, qubits3)
qc.ccs(qubits1, qubits2, qubits3)
qc.cct(qubits1, qubits2, qubits3)
qc.ccsdg(qubits1, qubits2, qubits3)
qc.cctdg(qubits1, qubits2, qubits3)
qc.ccrx(theta, qubits1, qubits2, qubits3)
qc.ccry(theta, qubits1, qubits2, qubits3)
qc.ccrz(theta, qubits1, qubits2, qubits3)
qc.ccp(phi, qubits1, qubits2, qubits3)
qc.cfsim(theta, phi, qubits1, qubits2, qubits3)
qc.ccu1(lamda, qubits1, qubits2, qubits3)
qc.ccu2(phi, lamda, qubits1, qubits2, qubits3)
qc.ccu3(theta, phi, lamda, qubits1, qubits2, qubits3)
qc.ccu(theta, phi, lamda, gamma, qubits1, qubits2, qubits3)

# show statevector
print(qc.state)
 ```

