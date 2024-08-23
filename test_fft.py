import numpy as np
from qutip import *
import scipy.integrate as integrate
from scipy.fft import fft
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt


def f(x):
    return np.cos(1 * x)

a = integrate.quad(f, 1, 2, complex_func=True)
N = 10000
t_list = np.linspace(-50, 50, N)
y_list = f(t_list)
c = fft(y_list)
w_list = np.linspace(0, 2*np.pi, N)
c = fftshift(c)
plt.plot(w_list, np.abs(c))
plt.show(block=True)
plt.plot(w_list, np.angle(c))
plt.show(block=True)

# ft = np.complex128(np.zeros(N))
# for i in range(0, N):
#     ker = y_list * np.exp(1j * w_list[i] * t_list)
#     ft[i] = integrate.simpson(ker, x=t_list)
#
# plt.plot(w_list, np.real(ft))
# plt.show(block=True)
