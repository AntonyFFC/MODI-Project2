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