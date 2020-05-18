import json
import pickle
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
from src.models.Hindmarsh_Rose_Neuron import Hindmarsh_Rose_Neuron
from src.models.Morris_Lecar_Neuron import Morris_Lecar_Neuron
from src.models.Hodgkin_Huxley_Neuron import Hodgkin_Huxley_Neuron
from src.models.Van_der_Pol_Oscillator import Van_der_Pol_Oscillator
from src.utils.limit_cycle_extraction import get_limit_cycle
from src.utils.phase_extraction import extract_protophase, extract_phase, get_period
from src.utils.utils import get_project_root
from cvxopt import matrix, solvers
from scipy.signal import resample, find_peaks
# generate data
# get frequency of oscillations
# get a couple of cycles at the end
# using the rhs of equation compute phase dependent sensitivity


def get_exact_PRC(Model, params, t_stop, num_resampling_points):
    M = Model(params)
    T_approx = get_period(M, t_stop)
    t, signal = get_limit_cycle(M, T_approx, num_resampling_points)
    T = t[-1]
    omega = (2 * np.pi) / (T)
    phase = t * omega
    # Using the rhs of equation compute phase dependent sensitivity from d phi /dt = omega = (Z, RHS(X))
    # representing the problem as min (x^T B^T @ B x) s.t Ax = b
    # matrix A corresponds to a constraint  (F(t), Z(t)) = omega
    N = signal.shape[1]
    L = signal.shape[0]
    A = np.zeros((L, L * N))
    b = omega * np.ones(L)
    # Matrix A compilation
    for i in range(L):
        A[i, i * N: (i + 1) * N] = M.rhs_(signal[i].reshape(-1, N)).squeeze()
    # Matrix B_1 compilation: [-1, 0, 1, 0] to extract coordinate of next vector from the previous
    # Matrix B_1 corresponds to a difference of PRCs at one time and the next time: Z(t+dt) - Z(t)
    tmp = np.zeros(2 * N)
    tmp[0] = -1
    tmp[N] = 1
    Tmp = np.vstack([np.roll(tmp, i) for i in range(N)])
    G = np.zeros((N, N * L))
    G[:N, : 2 * N] = Tmp
    B_1 = np.vstack([np.roll(G, 2 * i, axis = 1) for i in range(L)])
    # Matrix B_2 correponds to a difference of PRC at one time and the next time: [ Z(t+dt) - Z(t)) + dt * D(t)Z(t) ] (which should be 0)
    B_2 = deepcopy(B_1)
    for j in range(len(signal)):
        B_2[j * N : (j + 1) * N, j * N : (j + 1) * N] += dt * M.jac_rhs(signal[j].reshape(1, -1)).T
    B = np.vstack([B_1, B_2]) # Smoothness is much more important
    # Thus, matrix B corresponds to maximisation of smoothness of the PRCs and to minimisation of the dynamics violation

    # Posing an optimisation task
    alpha = 1e-9 #regularisation parameter
    P = matrix(2 * B.T @ B + alpha * np.eye(N * L))
    A = matrix(A)
    b = matrix(b)
    q = matrix(np.zeros(N * L))
    solvers.options['show_progress'] = True
    sol = solvers.qp(P=P, q=q, A=A, b=b)

    Z = np.array(sol['x']).reshape(L, N)
    phi = phase % (2 * np.pi)
    s = signal
    tmp = list(zip(phi, Z, s))
    tmp.sort(key=lambda a: a[0])
    phi, Z, s = zip(*tmp)
    Z = np.array(Z)
    s = np.array(s)
    prc_exact_data = dict()
    prc_exact_data["delta_phi"] = Z[:, 0]
    prc_exact_data["Z"] = Z
    prc_exact_data["phi"] = phi
    prc_exact_data["signal"] = s
    prc_exact_data["params"] = params
    prc_exact_data["omega"] = omega
    return prc_exact_data


if __name__ == '__main__':
    root_folder = get_project_root()
    # model_names = ['Hindmarsh_Rose_Neuron', 'Van_der_Pol_Oscillator', 'Morris_Lecar_Neuron', 'Hodgkin_Huxley_Neuron']
    model_names = ['Hindmarsh_Rose_Neuron', 'Van_der_Pol_Oscillator', 'Morris_Lecar_Neuron']
    for model_name in model_names:
        modifier = 'exact'
        tag = ''
        params = json.load(open(f"{root_folder}/data/model_params/{model_name}_params.json", 'r'))
        dt = params["dt"]
        t_stop = params["t_stop"]
        exact_prc_data = get_exact_PRC(eval(f"{model_name}"), params, t_stop, num_resampling_points=3000)
        name = f"{model_name}_{modifier}_{tag}_prc.pkl"
        pickle.dump(exact_prc_data, open(f"{root_folder}/data/processed_data/{name}", "wb+"))





