import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.linalg import toeplitz


# 2 a)
daneucz = np.loadtxt("danedynucz57.txt")
danewer = np.loadtxt("danedynwer57.txt")
u_ucz, y_ucz = daneucz[:,0], daneucz[:,1]
u_wer, y_wer = danewer[:,0], danewer[:,1]

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.plot(u_ucz, y_ucz, 'b.')
plt.title('Zbiór uczący')
plt.xlabel('Sygnał wejściowy (u)')
plt.xlabel('Sygnał wejściowy (y)')

plt.subplot(1, 2, 2)
plt.plot(u_wer, y_wer, 'r.')
plt.title('Zbiór weryfikujący')
plt.xlabel('Sygnał wejściowy (u)')
plt.ylabel('Sygnał wyjściowy (y)')

plt.tight_layout()
plt.show()

# b)

def fit_dynamic_model(u, y, nA, nB):
    N = len(y)

    # Tworzenie macierzy Toeplitza dla sygnałów u i y
    U = toeplitz(u, np.zeros(nB))
    Y = toeplitz(y, np.zeros(nA))

    # Przygotowanie macierzy regresji
    Phi = np.hstack([U[nB:], Y[nA:]])
    
    # Obcięcie wektora odpowiedzi
    y = y[max(nA, nB):]
    
    # Dopasowanie modelu
    theta = np.linalg.lstsq(Phi, y, rcond=None)[0]
    
    # Rozdzielenie współczynników
    b = theta[:nB]
    a = theta[nB:]
    
    return b, a