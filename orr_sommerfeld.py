#!/usr/bin/env python3

import numpy as np
from scipy.linalg import eigvals

# Gauss-Lobatto points

def gauss_lobatto_point(j, N):
    return np.cos(np.pi*j/N)

# Chebyshev polynomials

def chebyshev(y, n):
    if (n==0):
        return 1.0
    elif (n==1):
        return y
    else: 
        return 2*y*chebyshev(y, n-1) - chebyshev(y, n-2)

# Derivatives of chebyshev polynomials

def chebyshev_deriv(y, n, k):
    if (k==0):
        return chebyshev(y,n)
    else:
        if (n==0):
            return 0.0
        elif (n==1):
            return chebyshev_deriv(y, 0, k-1)
        elif (n==2):
            return 4.0*chebyshev_deriv(y, 1, k-1)
        else:
            return 2.0*n*chebyshev_deriv(y, n-1, k-1) + (n/(n-1.)*chebyshev_deriv(y, n-1, k))

# Define functions describing the flow feild

def L1(y, alpha, alp2, Re, iflow):
    
    # Poiseuille flow
    if(iflow == 1):
        return -alp2*(1-y**2) - (-2.0) - (alp2**2/(1j*alpha*Re))
    # Couette flow
    elif(iflow == 2):
        return -alp2*(y) - (alp2**2/(1j*alpha*Re))

def L2(y, alpha, alp2, Re, iflow):

    # Poiseuille flow
    if (iflow == 1):
        return (1-y**2) + 2*alp2/(1j*alpha*Re)
    # Couette flow
    elif(iflow == 2):
        return y + 2*alp2/(1j*alpha*Re)

# Solve Orr-Sommerfeld Equation

class Orr_Sommerfeld:
    
    # Constructor
    
    def __init__(self, N, alpha, beta, Re, iflow):
        
        self.N = N                                # No. of points in the domain
        self.alpha = alpha                        # Streamwise wavenumber 
        self.beta = beta                          # Spanwise wavenumber
        self.Re = Re                              # Reynolds number
        self.iflow = iflow                        # Type of flow: (Poiseuille=1, Couette=2)
        self.alp2 = self.alpha**2 + self.beta**2  # Parameter defined for convinience

        # For solving the eigenvalue problem Aq = cBq, define the matrices A and B

        self.A = np.zeros((self.N+1, self.N+1), dtype=complex)
        self.B = np.zeros((self.N+1, self.N+1))

        # Fill matrix B

        # Fill the rows corresponding to the boundary conditions 

        for j in range(0, self.N+1):
            # Top boundary 
            self.B[0,j] = chebyshev(1.0, j)             # Row 0 
            self.B[1,j] = chebyshev_deriv(1.0, j, 1)    # Row 1
            # Bottom boundary 
            self.B[N-1, j] = chebyshev(-1.0, j)         # Row N-1
            self.B[N,   j] = chebyshev_deriv(1.0, j, 1) # Row N
        
        # Fill the interiror rows

        for i in range(2, N-1):
            for j in range(0, N+1):
                y = gauss_lobatto_point(i, N)
                self.B[i,j] = chebyshev_deriv(y, j, 2) - self.alp2*chebyshev(y, j)

        # Fill matrix A

        # Fill the rows corresponding to the boundary conditions 

        for j in range(0, self.N+1):
            # Top boundary 
            self.A[0,j] = chebyshev(1.0, j)             # Row 0 
            self.A[1,j] = chebyshev_deriv(1.0, j, 1)    # Row 1
            # Bottom boundary 
            self.A[N-1, j] = chebyshev(-1.0, j)         # Row N-1
            self.A[N,   j] = chebyshev_deriv(1.0, j, 1) # Row N

        # Fill the interiror rows
        
        for i in range(2, N-1):
            for j in range(0, N+1):
                y = gauss_lobatto_point(i, N)
                term1 = L1(y, self.alpha, self.alp2, self.Re, self.iflow)*chebyshev(y, j)
                term2 = L2(y, self.alpha, self.alp2, self.Re, self.iflow)*chebyshev_deriv(y, j, 2)
                term3 = -chebyshev_deriv(y, j, 4)/(1j*self.alpha*self.Re)
                self.A[i,j] = term1 + term2 + term3

    def compute_eigen_values(self):
        
        # Compute the eigenvalues using scipy stack
        c = eigvals(self.A, self.B)
        return c