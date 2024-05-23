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

# Dopasowanie modelu liniowego metodą najmniejszych kwadratów
model1 = LinearRegression()
model1.fit(u_ucz.reshape(-1, 1), y_ucz)

a0, a1 = model1.intercept_, model1.coef_[0]

# Wyznaczenie wartości modelu
y_ucz_pred = model1.predict(u_ucz.reshape(-1, 1))
y_wer_pred = model1.predict(u_wer.reshape(-1, 1))

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

# Komentarz do wyników
print("Model liniowy został dopasowany do danych za pomocą metody najmniejszych kwadratów. "
      f"Błąd modelu dla zbioru uczącego wynosi {ucz_e:.4f}, natomiast dla zbioru weryfikującego wynosi {wer_e:.4f}. "
      "Jak widać na wykresach, model dobrze aproksymuje dane uczące, ale może mieć trudności z dokładnym "
      "dopasowaniem do danych weryfikujących, co sugeruje, że dane mogą mieć pewien stopień nieliniowości "
      "lub szumu.")

# c)

stopnie = [2, 5, 8, 11, 14]
ucz_errs = []
wer_errs = []

plt.figure(figsize=(18, 12))

for i, stopien in enumerate(stopnie):
    wielom = PolynomialFeatures(stopien)
    u_ucz_wiel = wielom.fit_transform(u_ucz.reshape(-1, 1))
    u_wer_wiel = wielom.fit_transform(u_wer.reshape(-1, 1))

    model2 = LinearRegression()
    model2.fit(u_ucz_wiel, y_ucz)

    y_ucz_pred_wiel = model2.predict(u_ucz_wiel)
    y_wer_pred_wiel = model2.predict(u_wer_wiel)

    ucz_errs.append(np.mean((y_ucz-y_ucz_pred_wiel) ** 2))
    wer_errs.append(np.mean((y_wer-y_wer_pred_wiel) ** 2))

    plt.subplot(3, 2, i+1)
    plt.plot(u_ucz, y_ucz, 'b.', label='Dane uczące')
    plt.plot(u_wer, y_wer, 'r.', label='Dane weryfikujące')
    u_range = np.linspace(min(u_ucz), max(u_ucz), 500)
    y_range = model2.predict(wielom.transform(u_range.reshape(-1, 1)))
    plt.plot(u_range, y_range, 'r-', label=f'Model wielomianowy N={stopien}')
    plt.title('Zbiór uczący')
    plt.xlabel('Sygnał wejściowy (u)')
    plt.xlabel('Sygnał wejściowy (y)')
    plt.title(f'Wielomian stopnia {stopien}')
    plt.legend()

plt.tight_layout()
plt.show()

import pandas as pd

errs_df = pd.DataFrame({
    'Stopień wielomianu': stopnie,
    'Błąd zbioru uczącego': ucz_errs,
    'Błąd zbioru weryfikującego': wer_errs
})

print(errs_df)