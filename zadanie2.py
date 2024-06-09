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

def create_M_matrix(u, y, nA, nB):
    N = len(y)
    M = np.zeros((N, nA + nB))
    
    for i in range(nB):
        M[:, i] = np.concatenate((np.zeros(i + 1), u[:N - i - 1]))
    
    for i in range(nA):
        M[:, nB + i] = np.concatenate((np.zeros(i + 1), y[:N - i - 1]))
    
    return M

def fit_model(u, y, nA, nB):
    M = create_M_matrix(u, y, nA, nB)
    M = M[max(nA, nB):]
    Y = y[max(nA, nB):]
    MTM_inv = np.linalg.inv(M.T @ M)
    MTY = M.T @ Y
    W = MTM_inv @ MTY
    return W

def predict_model(u, y, W, nA, nB, recursive=False):
    N = len(y)
    y_pred = np.zeros(N)
    
    for k in range(max(nA, nB), N):
        if recursive:
            y_k = y_pred
        else:
            y_k = y
            
        y_pred[k] = sum(W[i] * u[k - i - 1] for i in range(nB)) + sum(W[nB + i] * y_k[k - i - 1] for i in range(nA))
    
    return y_pred

def calculate_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


orders = [1, 2, 3]
results = []

for n in orders:
    W_ucz = fit_model(u_ucz, y_ucz, nA=n, nB=n)
    W_wer = fit_model(u_wer, y_wer, nA=n, nB=n)
    
    y_pred_ucz_nonrec = predict_model(u_ucz, y_ucz, W_ucz, nA=n, nB=n, recursive=False)
    y_pred_ucz_rec = predict_model(u_ucz, y_ucz, W_ucz, nA=n, nB=n, recursive=True)
    
    y_pred_wer_nonrec = predict_model(u_wer, y_wer, W_wer, nA=n, nB=n, recursive=False)
    y_pred_wer_rec = predict_model(u_wer, y_wer, W_wer, nA=n, nB=n, recursive=True)
    
    error_ucz_nonrec = calculate_error(y_ucz, y_pred_ucz_nonrec)
    error_ucz_rec = calculate_error(y_ucz, y_pred_ucz_rec)
    error_wer_nonrec = calculate_error(y_wer, y_pred_wer_nonrec)
    error_wer_rec = calculate_error(y_wer, y_pred_wer_rec)
    
    results.append((n, error_ucz_nonrec, error_ucz_rec, error_wer_nonrec, error_wer_rec))
    
    plt.figure(figsize=(12, 8))
    plt.suptitle(f'Model stopnia {n}', fontsize=15)

    plt.subplot(2, 2, 1)
    plt.plot(y_ucz, label='Prawdziwe')
    plt.plot(y_pred_ucz_nonrec, label='Predykcja')
    plt.title(f'Dane Uczące Brak Rekurencji')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(y_ucz, label='Prawdziwe')
    plt.plot(y_pred_ucz_rec, label='Predykcja')
    plt.title(f'Dane Uczące Rekurencja')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(y_wer, label='Prawdziwe')
    plt.plot(y_pred_wer_nonrec, label='Predykcja')
    plt.title(f'Dane Weryfikujące Brak Rekurencji')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(y_wer, label='Prawdziwe')
    plt.plot(y_pred_wer_rec, label='Predykcja')
    plt.title(f'Dane Weryfikujące Rekurencja')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Wyświetlenie wyników
import pandas as pd

df = pd.DataFrame(results, columns=['Rząd', 'Błąd Ucz B-Rek', 'Błąd Ucz Rek', 'Błąd Wer B-Rek', 'Błąd Wer Rek'])
print(df)

# Wybór najlepszego modelu
best_model = df.loc[df['Błąd Wer Rek'].idxmin()]
print(f'Najlepszy Model: Stopnia {best_model["Rząd"]} z błędem na danych weryfikujących z rekurencją: {best_model["Błąd Wer Rek"]}')