from qutip import *
import numpy as np
import matplotlib.pyplot as plt

times = np.linspace(0,10.0,200)
a = destroy(10)
x = a.dag() + a
H = a.dag() * a
corr1 = correlation_2op_1t(H, None, times, [np.sqrt(0.01) * a], x, x)
corr2 = correlation_2op_1t(H, None, times, [np.sqrt(1.0) * a], x, x)
corr3 = correlation_2op_1t(H, None, times, [np.sqrt(2.0) * a], x, x)
plt.figure()
plt.plot(times, np.real(corr1))
plt.plot(times, np.real(corr2))
plt.plot(times, np.real(corr3))
plt.legend(['0.5','1.0','2.0'])
plt.xlabel(r'Time $t$')
plt.ylabel(r'Correlation $\left<x(t)x(0)\right>$')
plt.show()