import json
import pickle
from scipy.signal import hilbert
import numpy as np
from collections import deque
from copy import deepcopy
from matplotlib import pyplot as plt
import numdifftools as nd
from src.utils.phase_extraction import extract_phase
from src.utils.utils import get_project_root

'''
 the class Van Der Pol Oscillator which computes time series, and accepts an input function
'''

class Van_der_Pol_Oscillator():
    def __init__(self, params):
        self.mu = params["mu"]
        self.dt = params["dt"]
        self.N = 2
        self.state = np.ones(self.N)
        self.noise_lvl = params["noise_lvl"]
        self.history_len = int(params["history_len"] / self.dt)
        self.input = 0
        self.state_history = deque(maxlen=self.history_len)
        self.t_range = deque(maxlen=self.history_len)
        self.input_history = deque(maxlen=self.history_len)
        self.state_history.append(deepcopy(self.state))
        self.t_range.append(0)
        self.input_history.append(deepcopy(self.input))


    def set_input(self, inp):
        self.input = inp
        return None

    def rhs_(self, state):
        rhs = np.zeros_like(state)
        x = state[0]
        y = state[1]
        rhs[0] = y + self.input
        rhs[1] = self.mu * (1 - x ** 2) * y - x
        return rhs.squeeze()

    # def jac_rhs(self, state):
    #     # dF_1(x,y)/dx dF_1(x,y)/dy
    #     # dF_2(x,y)/dx dF_2(x,y)/dy
    #     jac = np.zeros((self.N, self.N))
    #     x = state[0]
    #     y = state[1]
    #     jac[0, 0] = 0
    #     jac[0, 1] = 1
    #     jac[1, 0] = - 2 * self.mu * x * y - 1
    #     jac[1, 1] = self.mu * (1 - x ** 2)
    #     return jac

    def jac_rhs(self, state):
        return nd.Jacobian(self.rhs_)(state)

    def step_(self):
        '''
        rk4 difference scheme
        '''
        k_s1 = self.dt * self.rhs_(self.state)
        k_s2 = self.dt * self.rhs_(self.state + k_s1 / 2)
        k_s3 = self.dt * self.rhs_(self.state + k_s2 / 2)
        k_s4 = self.dt * self.rhs_(self.state + k_s3)
        new_state = self.state + 1.0 / 6.0 * (k_s1 + 2 * k_s2 + 2 * k_s3 + k_s4) + self.noise_lvl * np.random.randn(self.N)
        self.state = new_state
        return None

    def run(self, T):
        T_steps = int(np.ceil(T / self.dt))
        for i in range(T_steps):
            self.step_()
            self.state_history.append(deepcopy(self.state))
            self.input_history.append(deepcopy(self.input))
            self.t_range.append(self.t_range[-1] + self.dt)
        return None

    def reset(self):
        self.state_history = deque(maxlen=self.history_len)
        self.t_range = deque(maxlen=self.history_len)
        self.input_history = deque(maxlen=self.history_len)
        self.state_history.append(deepcopy(self.state))
        self.t_range.append(0)
        self.input_history.append(deepcopy(self.input))
        return None

    def plot_history(self):
        state_array = np.array(self.state_history) ###
        input_array = np.array(self.input_history)
        t_array = np.array(self.t_range)
        fig = plt.figure(figsize=(20, 10))
        plt.plot(t_array, state_array[:, 0], linewidth=3, color='k', label='amplitude')
        plt.plot(t_array, input_array, linewidth=3, color='r', label='input')
        s_an = hilbert(state_array[:, 0])
        protophase = np.angle(s_an - np.mean(s_an))
        phase = extract_phase(protophase, stim_start_ind = int(np.where(np.diff(input_array) > 0)[0]))
        plt.plot(t_array, phase % (2 * np.pi), linewidth=1, ls='--', color='g', label='phase')
        plt.legend(fontsize=24)
        plt.xlabel("time, ms", fontsize=24)
        plt.ylabel("Amplitude", fontsize=24)
        plt.grid(True)
        root_folder = get_project_root()
        plt.savefig(f"{root_folder}/img/Van_der_pol.png")
        plt.show(block=True)
        plt.close()
        return None

    # plot evoluton in phase plane?

if __name__ == '__main__':
    model_name = 'Van_der_Pol_Oscillator'
    root_folder = get_project_root()
    params = json.load(open(f"{root_folder}/data/model_params/{model_name}_params.json", 'r'))
    dt = params["dt"]
    t_stop = params["t_stop"]
    t_stop = 100
    stim_amp = 25
    t_stim_start = 55
    stim_duration = 2 * dt
    vdp = Van_der_Pol_Oscillator(params)
    vdp.run(t_stim_start)
    vdp.set_input(stim_amp)
    vdp.run(stim_duration)
    vdp.set_input(0)
    vdp.run(t_stop - (t_stim_start + stim_duration))
    vdp.plot_history()

    # saving data
    root_folder = get_project_root()
    data = dict()
    data['state'] = np.array(vdp.state_history)
    data['t'] = np.array(vdp.t_range).squeeze()
    data['inp'] = np.array(vdp.input_history)

    save_to = f"{root_folder}/data/runs/vdp_data.pkl"
    pickle.dump(data, open(save_to, "wb+"))







