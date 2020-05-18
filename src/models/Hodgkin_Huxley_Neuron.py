import json
import pickle
import numpy as np
from collections import deque
from copy import deepcopy
from matplotlib import pyplot as plt
from src.utils.utils import get_project_root
import numdifftools as nd

'''
 class Hodgkin Huxley Neuron which computes time series, and accepts an input function
'''
class Hodgkin_Huxley_Neuron():
    def __init__(self, params):
        self.C = params["C"]
        self.g_Na = params["g_Na"]
        self.V_Na = params["V_Na"]
        self.g_K = params["g_K"]
        self.V_K = params["V_K"]
        self.g_L = params["g_L"]
        self.V_L = params["V_L"]
        self.I0 = params["I0"]
        self.dt = params["dt"]
        self.noise_lvl = params["noise_lvl"]
        self.history_len = int(params["history_len"] / self.dt)
        self.N = 4
        self.state = np.ones(self.N)
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

    def alpha_m(self, state):
        v = state[0]
        return (2.5 - 0.1 * v) / (np.exp(2.5 - 0.1 * v) - 1)

    def beta_m(selfs, state):
        v = state[0]
        return 4 * np.exp(-v / 18)

    def alpha_h(self, state):
        v = state[0]
        return 0.07 * np.exp( - v / 20)

    def beta_h(selfs, state):
        v = state[0]
        return 1 /( np.exp(3 - 0.1 * v) + 1)

    def alpha_n(self, state):
        v = state[0]
        return (0.1 - 0.01 * v) / (np.exp(1 - 0.1 * v) - 1)

    def beta_n(selfs, state):
        v = state[0]
        return 0.125 * np.exp( -v / 80)

    def I_Na(self, state):
        v = state[0]
        m = state[1]
        h = state[2]
        n = state[3]
        return self.g_Na * m**3 * h * (v - self.V_Na)

    def I_K(self, state):
        v = state[0]
        n = state[3]
        return self.g_K * n ** 4 * (v - self.V_K)

    def I_L(self, state):
        v = state[0]
        return self.g_L * (v - self.V_L)

    def rhs_(self, state):
        rhs = np.zeros_like(state)
        v = state[0]
        m = state[1]
        h = state[2]
        n = state[3]
        rhs[0] = (1.0 / self.C) * (- self.I_Na(state) - self.I_K(state) - self.I_L(state) + self.I0 + self.input)
        rhs[1] = self.alpha_m(state) * (1 - m) - self.beta_m(state) * m
        rhs[2] = self.alpha_h(state) * (1 - h) - self.beta_h(state) * h
        rhs[3] = self.alpha_n(state) * (1 - n) - self.beta_n(state) * n
        return rhs

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
        state_array = np.array(self.state_history)
        input_array = np.array(self.input_history)
        t_array = np.array(self.t_range)
        fig = plt.figure(figsize=(20, 10))
        plt.plot(t_array, state_array[:, 0], linewidth=3, color='k', label='amplitude')
        plt.plot(t_array, input_array, linewidth=3, color='r', label='input')
        plt.legend(fontsize=24)
        plt.xlabel("time, ms", fontsize=24)
        plt.ylabel("Amplitude", fontsize=24)
        plt.grid(True)
        root_folder = get_project_root()
        plt.savefig(f"{root_folder}/img/Hodgkin_Huxley_Neuron.png")
        plt.show(block=True)
        plt.close()
        return None

    # plot evoluton in phase plane?

if __name__ == '__main__':
    model_name = 'Hodgkin_Huxley_Neuron'
    root_folder = get_project_root()
    params = json.load(open(f"{root_folder}/data/model_params/{model_name}_params.json", 'r'))
    dt = params["dt"]
    t_stop = params["t_stop"]
    stim_amp = 1
    t_stim_start = 350
    stim_duration = 10 * dt
    hhn = Hodgkin_Huxley_Neuron(params)
    hhn.run(t_stim_start)
    hhn.set_input(stim_amp)
    hhn.run(stim_duration)
    hhn.set_input(0)
    hhn.run(t_stop - (t_stim_start + stim_duration))
    hhn.plot_history()

    # saving data
    data = dict()
    data['state'] = np.array(hhn.state_history)
    data['t'] = np.array(hhn.t_range)
    data['inp'] = np.array(hhn.input_history)

    save_to = f"{root_folder}/data/runs/hhn_data.pkl"
    pickle.dump(data, open(save_to, "wb+"))







