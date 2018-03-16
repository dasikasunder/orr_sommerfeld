#!/usr/bin/env python3

from orr_sommerfeld import Orr_Sommerfeld

N = 5         # No. of modes
alpha = 2.1   # Streamwise wavenumber
beta = 2.1    # Spanwise wavenumber
Re = 10000    # Reynolds number
iflow = 1     # Type of flow: (Poiseuille=1, Couette=2)

test = Orr_Sommerfeld(N, alpha, beta, Re, iflow)
c = test.compute_eigen_values()
print(c)
