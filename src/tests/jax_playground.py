import numpy as np
from matplotlib import pyplot as plt
import sympy
from sympy import Matrix

def rhs(state):  # Define a function
    x = state[:, 0]
    y = state[:, 1]
    rhs = np.zeros(state.shape)
    rhs[:, 0] = y
    rhs[:, 1] = 4 * (1 - x ** 2) * y - x
    return rhs

def rhs_jac(state):
    dt = 1e-12
    return rhs(state) - rhs_




X = Matrix([rho*cos(phi), rho*sin(phi), rho**2])
Y = Matrix([rho, phi])

X.jacobian(Y)

