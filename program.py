import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1 a)
# Wczytanie danych statycznych
danestat = np.loadtxt("danestat57.txt")

# Podział danych na zbiór uczący i weryfikujący (50% każdy)
split_index = len(danestat)//2
danestatucz = danestat[:split_index]
danestatwer = danestat[split_index:]
u_ucz, y_ucz = danestatucz[:,0], danestatucz[:,1]
u_wer, y_wer = danestatwer[:,0], danestatwer[:,1]

# Narysowanie danych statycznych
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
# Obliczenie macierzy M
M = np.column_stack((np.ones(len(u_ucz)), u_ucz))

# Obliczenie macierzy Y
Y = y_ucz.reshape(-1, 1)

# Obliczenie wektorów współczynników W
MTM_inv = np.linalg.inv(M.T @ M)
MTY = M.T @ Y
W = MTM_inv @ MTY

b, a = W.flatten()

print("Współczynnik a:", a)
print("Współczynnik b:", b)

# Wyznaczenie modelu
def model(u):
    return b + a * u

y_ucz_pred = model(u_ucz)
y_wer_pred = model(u_wer)

# Obliczenie błędów modelu
ucz_e = np.mean((y_ucz-y_ucz_pred) ** 2)
wer_e = np.mean((y_wer-y_wer_pred) ** 2)

print(f"Błąd modelu dla zbioru uczącego: {ucz_e:.4f}")
print(f"Błąd modelu dla zbioru weryfikującego: {wer_e:.4f}")

# Narysowanie danych statycznych
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.plot(u_ucz, y_ucz, 'b.', label='Dane uczące')
plt.plot(u_ucz, y_ucz_pred, 'b-', label='Model liniowy')
plt.title('Zbiór uczący')
plt.xlabel('Sygnał wejściowy (u)')
plt.xlabel('Sygnał wejściowy (y)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(u_wer, y_wer, 'r.', label='Dane weryfikujące')
plt.plot(u_wer, y_wer_pred, 'r-', label='Model liniowy')
plt.title('Zbiór weryfikujący')
plt.xlabel('Sygnał wejściowy (u)')
plt.ylabel('Sygnał wyjściowy (y)')
plt.legend()

plt.tight_layout()
plt.show()
# c)

stopnie = [2, 5, 8, 11, 14]
ucz_errs = []
wer_errs = []

for i, stopien in enumerate(stopnie):

    # Obliczenie macierzy M
    M = np.ones((len(u_ucz), stopien + 1))
    for i in range(1, stopien + 1):
        M[:, i] = u_ucz ** i

    # Obliczenie macierzy Y
    Y = y_ucz.reshape(-1, 1)

    # Obliczenie wektorów współczynników W
    MTM_inv = np.linalg.inv(M.T @ M)
    MTY = M.T @ Y
    W = MTM_inv @ MTY

    coefficients = W.flatten()
    b = coefficients[0]
    a = coefficients[1:]

    print(f'Obliczone współczynniki: b = {b}, a = {a}')

    # Wyznaczenie modelu
    def model2(u):
        result = coefficients[0]
        for i in range(1, len(coefficients)):
            result += coefficients[i] * (u ** i)
        return result
    
    y_ucz_pred_wiel = model2(u_ucz)
    y_wer_pred_wiel = model2(u_wer)

    ucz_errs.append(np.mean((y_ucz-y_ucz_pred_wiel) ** 2))
    wer_errs.append(np.mean((y_wer-y_wer_pred_wiel) ** 2))

    u_range = np.linspace(min(u_ucz.min(), u_wer.min()), max(u_ucz.max(), u_wer.max()), 500)
    y_range_pred = model2(u_range)
    plt.figure(figsize=(10, 4))
    plt.scatter(u_ucz, y_ucz, color='blue', label='Dane uczące')
    plt.scatter(u_wer, y_wer, color='green', label='Dane weryfikujące')
    plt.plot(u_range, y_range_pred, color='red', label=f'Model nieliniowy')
    plt.title(f'stopien wielomianu={stopien}')
    plt.xlabel('u')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

import pandas as pd

errs_df = pd.DataFrame({
    'Stopień wielomianu': stopnie,
    'Błąd zbioru uczącego': ucz_errs,
    'Błąd zbioru weryfikującego': wer_errs
})

print(errs_df)