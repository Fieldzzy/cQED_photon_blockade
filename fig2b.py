import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from functions import *
from scipy.fft import fft
from scipy.fftpack import fftshift, fftfreq


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
    omgd = 1.153 * omg0
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

    t = np.linspace(0, 40, 200)
    rho0 = qt.tensor(qt.fock_dm(dim, 0), qt.fock_dm(2, 0))

    # L = qt.liouvillian(Hs)
    # result = qt.mesolve(L, rho0, t, [La, Ls], ad * a)
    # plt.plot(t, result.expect[0])
    # plt.show()

    L00 = qt.liouvillian(H0)
    L0 = L00 + Ls + La

    L_p = L_m = qt.liouvillian(0.5 * Omg * (a + ad))

    #  Original method calculating steady state, may inaccurate
    # Generate Sns and Tns
    # Calculate steady state
    # N = 10  # Maximum number of nonzero Sn
    # S = T = list(np.zeros(N+2))
    # S[N] = T[N] = qt.spre(qt.qzero([dim, 2]))
    # temp = range(0, N)
    # for i in reversed(temp):
    #     Temp_SuperS = L0 - 1j * (i+1) * omgd * qt.spre(qt.qeye([dim, 2])) + L_m * S[i+2]
    #     Temp_SuperT = L0 + 1j * (i+1) * omgd * qt.spre(qt.qeye([dim, 2])) + L_p * T[i+2]
    #     S[i+1] = - Temp_SuperS.inv() * L_p
    #     T[i+1] = - Temp_SuperT.inv() * L_m
    # Temp = L0 + L_m * S[1] + L_p * T[1]
    # rho_inf = Temp.inv() * qt.operator_to_vector(rho0)
    # rho_inf_dm = qt.vector_to_operator(rho_inf)
    # rho_ss = rho_inf_dm.unit()

    rho_ss = qt.steadystate_floquet(L00, [La, Ls], Omg * (a + ad), w_d=omgd)
    rho_ss = rho_ss.unit()

    # print((rho_ss - rho_inf_dm).norm())

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
    L_floquet = steadyoper_floquet(L00, [La, Ls], Omg * (a + ad), w_d=omgd)
    corr_list = qt.correlation_2op_1t(L0, rho_ss, tau_list, [La, Ls], X_dot_m, X_dot_p, reverse=False)
    prop = propagator(Hs, tau_list, [La, Ls])
    corr_list2 = np.zeros(points, complex)
    for i in range(0, points):
        corr_list2[i] = (rho_ss * X_dot_m * qt.vector_to_operator(prop[i].dag() * qt.operator_to_vector(X_dot_p))).tr()
    # plt.plot(tau_list, np.real(corr_list))
    # plt.show()

    # w_list, spec = qt.spectrum_correlation_fft(tau_list, corr_list)
    # figg, axx = plt.subplots(2, 1)
    # axx[0].plot(w_list, np.real(spec))
    # # axx[0].plot(w_list, (np.max(np.real(spec))/np.pi)*np.angle(spec))
    # axx[1].semilogy(w_list, np.abs(spec))
    # # axx[1].plot(w_list, (np.max(np.abs(spec))/np.pi)*np.angle(spec))
    # plt.show(block=True)
    return corr_list, corr_list2


def test():
    return np.sin(tau_list)


global tau_list
points = 10000
tau_range = 499.7
tau_list = np.linspace(0, tau_range, points)

corr_list, corr_list2 = main()
omg_list = fftshift(fftfreq(points, d=tau_range/points)) * 2 * np.pi
omg_list2 = np.linspace(omg0 * 0.2, omg0 * 1.7, points)
results = list(np.zeros(points))
for i in range(0, points):
    ker_list = corr_list2 * np.exp(1j * omg_list2[i] * tau_list)
    results[i] = 2 * integrate.simpson(ker_list, x=tau_list)
    print(i)

results = np.array(results)

w_list, spec1 = qt.spectrum_correlation_fft(tau_list, corr_list2, inverse=False)
spec2 = fftshift(fft(corr_list2))
fig, ax = plt.subplots(5, 1)
ax[0].plot(tau_list, np.abs(corr_list2))
ax[1].semilogy(omg_list2, np.abs(np.real(results)))
ax[2].plot(omg_list2, np.abs(results))
ax[3].semilogy(w_list, np.abs(spec1), marker='+')
ax[4].semilogy(omg_list, np.abs(np.real(spec2)))
plt.show(block=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
