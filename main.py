import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
from PIL import Image
import cv2


# def create_gif_and_video(total_time):
#     images_for_gif = []
#     images_for_video = []
#     for t in range(total_time):
#         images_for_gif.append(Image.open(f"images/plot_{t}.png"))

#         img = cv2.imread(f"images/plot_{t}.png")
#         height, width, layers = img.shape
#         size = (width, height)
#         images_for_video.append(img)

#     out = cv2.VideoWriter(
#         'project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

#     for i in range(len(images_for_video)):
#         out.write(images_for_video[i])
#     out.release()

#     images_for_gif[0].save('simulation.gif', format='GIF',
#                            append_images=images_for_gif[1:],
#                            save_all=True,
#                            duration=50, loop=0)


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

    delays = np.zeros((400, 31), dtype=np.int64)
    f = np.zeros_like(g)

    for transd in range(x.size):
        for zz in range(z.size):
            for xx in range(x.size):
                delays[zz, xx] = round(((math.sqrt(
                    (z[zz, 0] - z[0, 0]) ** 2 + (x[xx, 0] - x[transd, 0]) ** 2) * 2) / cl) / T)
                if delays[zz, xx] <= 399:
                    f[zz, xx] += g[delays[zz, xx], transd]
        if transd == 0:
            plt.figure()
            plt.imshow(f, aspect='auto')
            plt.title('Primeira aquisição')
            plt.show()

    return f


f = saft(g, x, z, cl, T)
plt.figure()
plt.imshow(f, aspect='auto')
plt.title('SAFT')
plt.show()

# create_gif_and_video(x.size)
