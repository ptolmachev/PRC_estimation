import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
x = np.linspace(-2, 2, 100)


def vect_fun(state):
    rhs = np.zeros_like(state)
    x = state[0]
    y = state[1]
    rhs[0] = y
    rhs[1] = 4 * (1 - x ** 2) * y - x
    return rhs.squeeze()

def jac_vect_fun(states):
    jac = np.zeros((2, 2))
    x = states[0]
    y = states[1]
    jac[0, 0] = 0
    jac[0, 1] = 1
    jac[1, 0] = - 2 * 4 * x * y - 1
    jac[1, 1] = 4 * (1 - x ** 2)
    return jac

jac = nd.Jacobian(vect_fun)
states = np.random.rand(2)

print(jac(states))
print(jac_vect_fun(states))



