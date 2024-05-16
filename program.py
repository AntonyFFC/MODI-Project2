import numpy as np
import matplotlib.pyplot as plt

# Wczytanie danych statycznych
danestat = np.loadtxt("danestat57.txt")

# Podział danych na zbiór uczący i weryfikujący (50% każdy)
split_index = len(danestat)//2
danestatucz = danestat[:split_index]
danestatwer = danestat[split_index:]

# Wyodrębnienie wejścia i wyjścia
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