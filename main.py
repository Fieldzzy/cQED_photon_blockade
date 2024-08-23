import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

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


def main(omgd):
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

    g20 = (rho_ss * X_dot_m * X_dot_m * X_dot_p * X_dot_p).tr() / ((rho_ss * X_dot_m * X_dot_p).tr()) ** 2
    return np.abs(g20)


points = 400
omgds = np.linspace(0.6 * omg0, 1.4 * omg0, points)
results = list(np.zeros(points))
for i in range(0, points):
    results[i] = main(omgds[i])
    print(i)

results = np.array(results)
fig, ax = plt.subplots()
ax.semilogy(omgds, results)
plt.show(block=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
