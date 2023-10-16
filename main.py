import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math

file = "DadosEnsaio.mat"

mat = scipy.io.loadmat(file)

pi = 500  # Amostra inicial dos A-Scan
pf = 900  # Amostra final dos A-Scan
g = mat["ptAco40dB_1"]["AscanValues"][0][0][pi:pf]  # B-Scan
cl = mat["ptAco40dB_1"]["CscanData"][0][0]["Cl"][0][0][0][0]  # Velocidade
t = mat["ptAco40dB_1"]["timeScale"][0][0][pi:pf]*1e-6  # Tempo
T = t[1][0]-t[0][0]  # Período de amostragem
z = cl*t/2  # Conversação para posição /2->ida/volta
x = mat["ptAco40dB_1"]["CscanData"][0][0]["X"][0][0]*1e-3  # Posições transdut

plt.figure()
plt.imshow(g, aspect="auto")
plt.title('B-Scan')
plt.show()


def saft(g, x, z, cl, T):

    delays = np.zeros_like(g, dtype=np.int64)
    f = np.zeros_like(g)

    for transd in range(x.size):
        for zz in range(z.size):
            for xx in range(x.size):
                r = math.sqrt(z[zz, 0] ** 2 + (x[xx, 0] -
                              x[transd, 0]) ** 2) - z[0, 0]
                delays[zz, xx] = round(r * 2 / cl / T)
                if delays[zz, xx] <= 399:
                    f[zz, xx] += g[delays[zz, xx], transd]

    return f


f = saft(g, x, z, cl, T)
plt.figure()
plt.imshow(f, aspect='auto')
plt.title('SAFT')
plt.show()
