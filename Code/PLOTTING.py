import numpy as np
import matplotlib.pyplot as plt
# -------------PLOTTING FUNCTIONS----------------------
size=(14,7)
def imshow(f1, f2, name, e0, d0, C, n, path='', headline=''):
    plt.figure(figsize=size)
    plt.subplot(121)
    plt.imshow(np.transpose(f1), origin='lower')
    plt.colorbar()
    plt.title(headline[0], fontsize=15)
    plt.subplot(122)
    plt.imshow(np.transpose(f2), origin='lower')
    plt.colorbar()
    plt.title(headline[1], fontsize=15)
    plt.savefig(path+name, dpi=300)

