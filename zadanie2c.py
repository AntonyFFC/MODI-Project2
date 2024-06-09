import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.linalg import toeplitz

daneucz = np.loadtxt("danedynucz57.txt")
danewer = np.loadtxt("danedynwer57.txt")
u_ucz, y_ucz = daneucz[:,0], daneucz[:,1]
u_wer, y_wer = danewer[:,0], danewer[:,1]

chosen_N = 4
chosen_degree = 5

def create_M_matrix(u, y, nA, nB, degree):
    N = len(y)
    M = np.zeros((N, (nA + nB) * degree))
    
    for i in range(1, nA + 1):
        for d in range(1, degree + 1):
            M[i:, (i - 1) * degree + (d - 1)] = y[:-i] ** d
            
    for j in range(1, nB + 1):
        for d in range(1, degree + 1):
            M[j:, nA * degree + (j - 1) * degree + (d - 1)] = u[:-j] ** d
    
    return M

def fit_model(u, y, nA, nB, degree):
    M = create_M_matrix(u, y, nA, nB, degree)
    M = M[max(nA, nB):]
    Y = y[max(nA, nB):]
    MTM_inv = np.linalg.inv(M.T @ M)
    MTY = M.T @ Y
    W = MTM_inv @ MTY
    return W

def predict_model(u, y, w, nA, nB, degree, recursive=False):
    N = len(y)
    y_pred = np.zeros(N)  

    for k in range(max(nA, nB), N):
        if recursive:
            y_k = y_pred
        else:
            y_k = y
        y_sum = 0
        for i in range(1, nA + 1):
            for d in range(1, degree + 1):
                y_sum += w[(i - 1) * degree + (d - 1)] * ((y_k[k - i]) ** d)               
        u_sum = 0
        for j in range(1, nB + 1):
            for d in range(1, degree + 1):
                u_sum += w[nA * degree + (j - 1) * degree + (d - 1)] * (u[k - j] ** d)              
        y_pred[k] = y_sum + u_sum
    return y_pred

def calculate_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
model_configs = [(nA, nA, degree) for nA in range(1, 9) for degree in range(1, 6)]
results = []

for nA, nB, degree in model_configs:
    w = fit_model(u_ucz, y_ucz, nA, nB, degree)
    
    y_ucz_pred = predict_model(u_ucz, y_ucz, w, nA, nB, degree, recursive=False)
    y_wer_pred = predict_model(u_wer, y_wer, w, nA, nB, degree, recursive=False)
    
    y_ucz_pred_recurrent = predict_model(u_ucz, y_ucz, w, nA, nB, degree, recursive=True)
    y_wer_pred_recurrent = predict_model(u_wer, y_wer, w, nA, nB, degree, recursive=True)
    
    ucz_errs = calculate_error(y_ucz, y_ucz_pred)
    wer_errs = calculate_error(y_wer, y_wer_pred)
    ucz_errs_rec = calculate_error(y_ucz, y_ucz_pred_recurrent)
    wer_errs_rec = calculate_error(y_wer, y_wer_pred_recurrent)
    
    results.append((nA, nB, degree, ucz_errs, wer_errs, ucz_errs_rec, wer_errs_rec))

def plot_model_results(u, y_true, y_pred, y_recurrent, title1, title2, TYPE):
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.scatter(range(len(y_true)), y_true, label='Dane', color='red', marker='o', s=10)
    plt.plot(y_pred, label='Model', color='blue')
    plt.title(title1)
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.scatter(range(len(y_true)), y_true, label='Dane', color='red', marker='o', s=10)
    plt.plot(y_recurrent, label='Model', color='blue')
    plt.title(title2)
    plt.legend()
    plt.grid(True)
    plt.show()

import pandas as pd

df = pd.DataFrame(results, columns=['nA','nB','Stopień', 'Błąd Ucz B-Rek', 'Błąd Wer B-Rek', 'Błąd Ucz Rek', 'Błąd Wer Rek'])
print(df)

for result in results:
    if result[0] == chosen_N and result[2] == chosen_degree:
#         print(f"Model results: nA={result['nA']}, nB={result['nB']}, degree={result['degree']}:")
#         print(f"ucz_errs (non-recursive): {round(result['ucz_errs'], 3)}")
#         print(f"wer_errs (non-recursive): {round(result['wer_errs'], 3)}")
#         print(f"ucz_errs (recursive): {round(result['ucz_errs_rec'], 3)}")
#         print(f"wer_errs (recursive): {round(result['wer_errs_rec'], 3)}")
        
        w = fit_model(u_ucz, y_ucz, chosen_N, chosen_N, chosen_degree)
        y_ucz_pred = predict_model(u_ucz, y_ucz, w, chosen_N, chosen_N, chosen_degree, recursive=False)
        y_ucz_pred_recurrent = predict_model(u_ucz, y_ucz, w, chosen_N, chosen_N, chosen_degree, recursive=True)
        
        plot_model_results(u_ucz, y_ucz, y_ucz_pred, y_ucz_pred_recurrent, f'Dane uczące się - Nierekurencyjny (nA={chosen_N}, stopień={chosen_degree},)', f'Dane uczące się - Rekurencyjny (nA={chosen_N}, stopień={chosen_degree}', 'train')

        y_wer_pred = predict_model(u_wer, y_wer, w, chosen_N, chosen_N, chosen_degree, recursive=False)
        y_wer_pred_recurrent = predict_model(u_wer, y_wer, w, chosen_N, chosen_N, chosen_degree, recursive=True)
        
        plot_model_results(u_wer, y_wer, y_wer_pred, y_wer_pred_recurrent, f'Dane weryfikujące - Nierekurencyjny (nA={chosen_N}, stopień={chosen_degree})', f'Dane weryfikujące - Rekurencyjny (nA={chosen_N}, stopień={chosen_degree})', 'test')

# print()
# print("Wyniki dla wszystkich modeli:")
# for result in results:
#     print(f"nA={result['nA']}, nB={result['nB']}, degree={result['degree']}:")
#     print(f"ucz_errs (non-recursive): {round(result['ucz_errs'], 3)}")
#     print(f"wer_errs (non-recursive): {round(result['wer_errs'], 3)}")
#     print(f"ucz_errs (recursive): {round(result['ucz_errs_rec'], 3)}")
#     print(f"wer_errs (recursive): {round(result['wer_errs_rec'], 3)}")
#     print()

def static_characteristic(N: int, K: int, recursive: bool):
    u = np.linspace(-1, 1, 2000)
    w = fit_model(u_ucz, y_ucz, N, N, K)
    y = list(danewer[:N, 1])
    for i in range(N-1, len(u)-1):
        row = []
        for j in range(N):
            for k in range(1, K + 1):
                row.append(u[i - j] ** k)
                row.append(y[i - j] ** k)     
        temp = 0
        for l in range(len(row)):
            temp += w[l] * row[l]
        y.append(temp)
    plt.figure
    plt.plot(u[:45], y[:45], c='y', linewidth=1.5)
    plt.plot(u[45:], y[45:], c='b', linewidth=1.5)
    plt.xlabel("u")
    plt.ylabel("y")
    plt.title("Charakterystyka statyczna na podstawie najlepszego modelu dynamicznego")
    # plt.savefig(f'/home/adrian/MODI/plots/zad2d_charakt.png')
    plt.show()
static_characteristic(chosen_N, chosen_degree, True)