import numpy as np
from qutip import *
import matplotlib.pyplot as plt

times = np.linspace(0.0, 10.0, 100)
H = 2*np.pi * 0.1 * sigmax()
psi0 = basis(2, 0)
times = np.linspace(0.0, 10.0, 100)
# result = mesolve(H, psi0, times, [np.sqrt(0.05) * sigmax()], e_ops=[sigmaz(), sigmay()])
L0 = liouvillian(H)
Lc = lindblad_dissipator(np.sqrt(0.05) * sigmax())
L = L0 + Lc
result = mesolve(L0, psi0, times, [Lc], e_ops=[sigmaz(), sigmay()])
result2 = propagator(H, 1, c_ops=sigmax())
fig, ax = plt.subplots()
ax.plot(times, result.expect[0])
ax.plot(times, result.expect[1])
ax.set_xlabel('Time')
ax.set_ylabel('Expectation values')
ax.legend(("Sigma-Z", "Sigma-Y"))
plt.show()