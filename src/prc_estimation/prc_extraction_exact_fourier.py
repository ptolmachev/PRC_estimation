import json
import pickle
import numpy as np
from matplotlib import pyplot as plt
from src.models.Hindmarsh_Rose_Neuron import Hindmarsh_Rose_Neuron
from src.models.Morris_Lecar_Neuron import Morris_Lecar_Neuron
from src.models.Hodgkin_Huxley_Neuron import Hodgkin_Huxley_Neuron
from src.models.Van_der_Pol_Oscillator import Van_der_Pol_Oscillator
from cvxopt import matrix, solvers
from src.utils.limit_cycle_extraction import get_limit_cycle
from src.utils.phase_extraction import get_period
from copy import deepcopy
from src.utils.utils import get_project_root

# estimation of exact PRC using Fourier decomposition

def get_exact_PRC(Model, params, t_stop, num_resampling_points, N_fourirer_components):
    model = Model(params)
    T_approx = get_period(model, t_stop)
    t, signal = get_limit_cycle(model, T_approx, num_resampling_points)
    T = t[-1]
    omega = (2 * np.pi) / (T)
    phase = t * omega

    N = signal.shape[1]
    M = signal.shape[0] # time points
    K = N_fourirer_components * 2 + 1

    C = np.zeros((M, N * K)) # d phi/ dt = (F(t), Z(t)) = w
    c = omega * np.ones(M)
    for t_i in range(M):
        F_t_i = deepcopy(model.rhs_(signal[t_i, :].reshape(1, N))).squeeze()
        F_row = np.hstack([F_t_i for i in range(K)])
        cos = np.array([np.cos((k + 1) * omega * t[t_i]) for k in range(N_fourirer_components)])
        sin = np.array([np.sin((k + 1) * omega * t[t_i]) for k in range(N_fourirer_components)])
        cos_N = np.kron(cos, np.ones(N))
        sin_N = np.kron(sin, np.ones(N))
        Fourier_row = np.hstack([np.ones(N), cos_N, sin_N])
        full_row_t_i = F_row * Fourier_row
        C[t_i, :] = deepcopy(full_row_t_i)

    G = np.zeros((M * N, N * K)) # dZ/dt + D(t) Z = 0; Adjoint equation
    g = np.zeros(M * N)
    for t_i in range(M):
        D_t_i = deepcopy(model.jac_rhs(signal[t_i, :].reshape(1, N)).T)
        D_row =  np.hstack([D_t_i for k in range(K)])
        I_row = np.hstack([np.eye(N) for k in range(K)])
        cos = np.array([np.cos((k + 1) * omega * t[t_i]) for k in range(N_fourirer_components)])
        sin = np.array([np.sin((k + 1) * omega * t[t_i]) for k in range(N_fourirer_components)])
        cos_N = np.kron(cos, np.ones(N))
        sin_N = np.kron(sin, np.ones(N))
        kw_cos = np.array([(k + 1) * omega * np.cos((k + 1) * omega * t[t_i]) for k in range(N_fourirer_components)])
        kw_sin = np.array([(k + 1) * omega * np.sin((k + 1) * omega * t[t_i]) for k in range(N_fourirer_components)])
        kw_cos_N = np.kron(kw_cos, np.ones(N))
        kw_sin_N = np.kron(kw_sin, np.ones(N))
        Row_part_1 = D_row * np.hstack([np.ones(N), cos_N, sin_N])
        Row_part_2 =  I_row * np.hstack([np.zeros(N), -kw_sin_N, kw_cos_N])
        full_row_t_i = Row_part_1 + Row_part_2
        G[t_i * N: (t_i + 1) * N, :] = deepcopy(full_row_t_i)

    # Least norm solution (no hard constraints) using solver
    alpha = 1e-9
    S = np.vstack([1000 * C, G]) # the constraint (F(t), Z(t)) = w are much more important
    s = np.hstack([1000 * c, g])
    q = matrix((-2 * s.reshape(1, -1) @ S).squeeze())
    solvers.options['show_progress'] = True
    sol = solvers.qp(P=matrix(2 * S.T @ S + alpha * (np.eye(N * K))), q=q)
    x = np.array(sol['x'])

    coeffs = x.reshape(K, N)
    cos = np.hstack([np.cos((k + 1) * omega * t).reshape(-1, 1) for k in range(N_fourirer_components)]) # M x N_fourier_comp
    sin = np.hstack([np.sin((k + 1) * omega * t).reshape(-1, 1) for k in range(N_fourirer_components)]) # M x N_fourier_comp
    A = np.hstack([np.ones((M, 1)), cos, sin]) # M x K
    Z = A @ coeffs
    prc_exact_data = dict()
    prc_exact_data["delta_phi"] = Z[:, 0]
    prc_exact_data["Z"] = Z
    prc_exact_data["phi"] = phase
    prc_exact_data["signal"] = signal
    prc_exact_data["params"] = params
    prc_exact_data["omega"] = omega
    return prc_exact_data


if __name__ == '__main__':
    root_folder = get_project_root()
    model_names = ['Van_der_Pol_Oscillator', 'Morris_Lecar_Neuron',  'Hindmarsh_Rose_Neuron']
    # model_names = ['Hindmarsh_Rose_Neuron']
    for model_name in model_names:
        modifier = 'exact_fourier'
        tag = ''
        params = json.load(open(f"{root_folder}/data/model_params/{model_name}_params.json", 'r'))
        dt = params["dt"]
        t_stop = params["t_stop"]
        exact_prc_data = get_exact_PRC(eval(f"{model_name}"), params, t_stop, num_resampling_points=5000, N_fourirer_components=1000)
        name = f"{model_name}_{modifier}_{tag}_prc.pkl"
        pickle.dump(exact_prc_data, open(f"{root_folder}/data/processed_data/{name}", "wb+"))