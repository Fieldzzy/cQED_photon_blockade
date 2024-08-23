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


def main(g):
    H0 = omg0 * ad * a + omgx * sigmap * sigmam + g * (a + ad) * (np.cos(theta) * sigmaz - np.sin(theta) * sigmax)
    eigens = H0.eigenstates()
    eigenvalues = eigens[0]
    eigenstates = eigens[1]
    return eigenvalues


points = 100
g_list = np.linspace(0 * omg0, 0.4 * omg0, points)
results = list(np.zeros(points))
for i in range(0, points):
    results[i] = main(g_list[i])
    print(i)

results = np.array(results)
fig, ax = plt.subplots()
ax.plot(g_list, results)
plt.show(block=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
