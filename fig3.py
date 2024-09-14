import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from functions import *

omg0 = 1
omgx = omg0
Omg = 1E-4 * omg0
theta = 0.93
dim = 5
gamma_a = 1E-2 * omg0
gamma_s = 1E-2 * omg0
X0 = 10
g = 0.2 * omg0

a = qt.tensor(qt.destroy(dim), qt.qeye(2))
ad = a.dag()
sigmap = qt.tensor(qt.qeye(dim), qt.sigmap())
sigmam = sigmap.dag()
sigmax = qt.tensor(qt.qeye(dim), qt.sigmax())
sigmaz = qt.tensor(qt.qeye(dim), qt.sigmaz())


def main():
    # |0> - |1> : 0.844, - |2> : 1.153, -|3> : 1.787
    omgd = 1.15 * omg0
    H0 = omg0 * ad * a + omgx * sigmap * sigmam + g * (a + ad) * (np.cos(theta) * sigmaz - np.sin(theta) * sigmax)
    Hdrive = qt.QobjEvo([[Omg * a, lambda t: np.cos(omgd * t)], [Omg * ad, lambda t: np.cos(omgd * t)]])
    Hs = H0 + Hdrive
    eigens = H0.eigenstates()
    eigenvalues = eigens[0]
    eigenstates = eigens[1]

    # Generate collapse superoperators
    La = Ls = qt.to_super(qt.qzero([dim, 2]))
    for j in range(0, len(eigenvalues)):
        for k in range(0, len(eigenvalues)):
            if eigenvalues[k] > eigenvalues[j]:
                # Generate transition coefficient
                A = -1j * eigenstates[j].dag() * (a - ad) * eigenstates[k]
                S = -1j * eigenstates[j].dag() * (sigmam - sigmap) * eigenstates[k]
                # Generate relaxation coefficient
                Gamma_a = gamma_a * ((eigenvalues[k] - eigenvalues[j]) / omg0) * A * A.conjugate()
                Gamma_s = gamma_s * ((eigenvalues[k] - eigenvalues[j]) / omg0) * S * S.conjugate()
                La = La + Gamma_a * qt.lindblad_dissipator(eigenstates[j] * eigenstates[k].dag())
                Ls = Ls + Gamma_s * qt.lindblad_dissipator(eigenstates[j] * eigenstates[k].dag())


    L00 = qt.liouvillian(H0)
    L0 = L00 + Ls + La

    rho_ss = qt.steadystate_floquet(L00, [La, Ls], Omg * (a + ad), w_d=omgd)
    rho_ss = rho_ss.unit()

    # Calculate positive and negative components of X
    X = -1j * X0 * (a - ad)
    X_dot_p = X_dot_m = qt.qzero([dim, 2])
    for j in range(0, len(eigenvalues)):
        for k in range(0, len(eigenvalues)):
            if eigenvalues[k] > eigenvalues[j]:
                X_dot_p = X_dot_p + (-1j) * (eigenvalues[k] - eigenvalues[j]) * (
                        eigenstates[j].dag() * X * eigenstates[k]) \
                          * eigenstates[j] * eigenstates[k].dag()
                X_dot_m = X_dot_p.dag()

    # g20 = (rho_ss * X_dot_m * X_dot_m * X_dot_p * X_dot_p).tr() / ((rho_ss * X_dot_m * X_dot_p).tr()) ** 2

    prop = propagator(Hs, tau_list, [La, Ls])
    g2a = np.zeros(points, complex)
    for i in range(0, points):
        g2a[i] = (rho_ss * X_dot_m * qt.vector_to_operator(prop[i].dag() * qt.operator_to_vector(X_dot_m))\
                 * qt.vector_to_operator(prop[i].dag() * qt.operator_to_vector(X_dot_p)) * X_dot_p).tr() \
                / ((rho_ss * X_dot_m * X_dot_p).tr()) ** 2

    g2b = qt.correlation_3op_1t(L00, rho_ss, tau_list, [La, Ls], X_dot_m, X_dot_m * X_dot_p, X_dot_p)
    g2b = g2b / ((rho_ss * X_dot_m * X_dot_p).tr()) ** 2
    return g2a, g2b


global tau_list
points = 2000
tau_range = 1000
tau_list = np.linspace(0, tau_range, points)
g2a, g2b = main()

fig, ax = plt.subplots(2, 1)
ax[0].semilogy(tau_list, np.abs(g2a))
ax[1].semilogy(tau_list, np.abs(g2b))
plt.show(block=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
