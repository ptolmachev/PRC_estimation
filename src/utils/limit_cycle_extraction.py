import numpy as np
import sympy
from matplotlib import pyplot as plt
from scipy.integrate import odeint, solve_bvp
from src.models.Hindmarsh_Rose_Neuron import Hindmarsh_Rose_Neuron
from src.models.Van_der_Pol_Oscillator import Van_der_Pol_Oscillator
from src.utils.phase_extraction import get_period
from src.utils.utils import get_project_root
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

def get_limit_cycle(Model, T, num_points):
    # 3) run the model once again with dt = T / (1000 - 1)
    dt = T / (50000) # so that there is exactly 50000 points for each cycle
    Model.history_len = 300000
    Model.reset()
    Model.dt = dt
    Model.run(T * 10) #10 cycles
    states = np.array(Model.state_history)
    t = np.array(Model.t_range)
    signal = states
    omega = 2 * np.pi / T
    phase = t * omega
    inds = np.argwhere(np.diff((phase) % (2 * np.pi)) <= -np.pi).squeeze()
    # We take only three cycles close to the end when the transients are gone:
    cycles_start = inds[-4]
    cycles_finish = inds[-4 + 3] - 1
    signal_ = signal[cycles_start: cycles_finish]
    t_ = t[cycles_start: cycles_finish]
    r = np.sum(((signal_[0, :] - signal_) ** 2), axis=1)
    inds = find_peaks(-r, height=-0.002)[0]
    inds_per_cycle = int(np.mean(np.diff(inds)))
    signal__ = signal_[:inds_per_cycle]
    signal__[0, :] = signal__[-1, :]
    t__ = np.arange(inds_per_cycle) * dt
    cs = CubicSpline(t__, signal__, bc_type='periodic')
    t = np.linspace(0, t__[-1], num_points)
    return t, cs(t)

