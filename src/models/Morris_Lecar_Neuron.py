import json
import pickle
import numpy as np
from collections import deque
from copy import deepcopy
from matplotlib import pyplot as plt
from src.utils.utils import get_project_root
import numdifftools as nd

'''
 class Morris Lecar Neuron which computes time series, and accepts an input function
'''
class Morris_Lecar_Neuron():
    def __init__(self, params):
        self.C = params["C"]
        self.g_Ca = params["g_Ca"]
        self.V_Ca = params["V_Ca"]
        self.g_K = params["g_K"]
        self.V_K = params["V_K"]
        self.g_L = params["g_L"]
        self.V_L = params["V_L"]
        self.V1 = params["V1"]
        self.V2 = params["V2"]
        self.V3 = params["V3"]
        self.V4 = params["V4"]
        self.phi = params["phi"]
        self.I0 = params["I0"]
        self.dt = params["dt"]
        self.N = 2
        self.noise_lvl = params["noise_lvl"]
        self.history_len = int(params["history_len"] / self.dt)
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

    def m_inf(self, v):
        return 0.5 * (1 + np.tanh((v - self.V1) / (self.V2)))

    def w_inf(self, v):
        return 0.5 * (1 + np.tanh((v - self.V3) / (self.V4)))

    def tau_w(self, v):
        return 1.0 / (np.cosh((v - self.V3)/(2 * self.V4)))

    def I_Ca(self, state):
        v = state[0]
        w = state[1]
        return self.g_Ca * self.m_inf(v) * (v - self.V_Ca)

    def I_K(self, state):
        v = state[0]
        w = state[1]
        return self.g_K * w * (v - self.V_K)

    def I_L(self, state):
        v = state[0]
        w = state[1]
        return self.g_L * (v - self.V_L)

    def rhs_(self, state):
        rhs = np.zeros_like(state)
        rhs[0] = (1.0 / self.C) * (- self.I_Ca(state) - self.I_K(state) - self.I_L(state) + self.I0 + self.input)
        rhs[1] = self.phi * (self.w_inf(state[0]) - state[1]) / (self.tau_w(state[0]))
        return rhs

    def jac_rhs(self, state):
        return nd.Jacobian(self.rhs_)(state)

    # def jac_rhs(self, state):
    #     # dF_1(x,y)/dx dF_1(x,y)/dy
    #     # dF_2(x,y)/dx dF_2(x,y)/dy
    #     jac = np.zeros((self.N, self.N))
    #     v = state[0]
    #     w = state[1]
    #
    #     def d_I_Ca_dv(state):
    #         v = state[0]
    #         w = state[1]
    #         return (self.g_Ca * self.m_inf(v) + self.g_Ca * (1/(self.V2)) * self.m_inf(v) * (1 - self.m_inf(v)) * (v - self.V_Ca))
    #
    #     def d_I_K_dv(state):
    #         v = state[0]
    #         w = state[1]
    #         return self.g_K * w
    #
    #     def d_I_K_dw(state):
    #         v = state[0]
    #         w = state[1]
    #         return self.g_K * (v - self.V_K)
    #
    #     def d_w_inf_dv(v):
    #         return (1/(self.V4)) * self.w_inf(v) * (1 - self.w_inf(v))
    #
    #     def d_tau_w_dv(v):
    #         return -(1/(2 * self.V4)) * np.sinh((v - self.V3)/(2 * self.V4)) / (np.cosh((v - self.V3)/(2 * self.V4)))**2
    #
    #     jac[0, 0] = (1.0 / self.C) * (-d_I_Ca_dv(state) - d_I_K_dv(state) - self.g_L)
    #     jac[0, 1] = (1.0 / self.C) * (-d_I_K_dw(state))
    #     jac[1, 0] = self.phi * (d_w_inf_dv(v) * self.tau_w(v) - self.w_inf(v) * d_tau_w_dv(v)) / self.tau_w(v)**2
    #     jac[1, 1] = - self.phi / (self.tau_w(state[:, 0]))
    #     return jac

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
        plt.savefig(f"{root_folder}/img/Morris_Lecar_Neuron.png")
        plt.show(block=True)
        plt.close()
        return None

    # plot evoluton in phase plane?

if __name__ == '__main__':
    model_name = 'Morris_Lecar_Neuron'
    root_folder = get_project_root()
    params = json.load(open(f"{root_folder}/data/model_params/{model_name}_params.json", 'r'))
    dt = params["dt"]
    t_stop = params["t_stop"]
    stim_amp = 10
    t_stim_start = 350
    stim_duration = 100 * dt
    mln = Morris_Lecar_Neuron(params)
    mln.run(t_stim_start)
    mln.set_input(stim_amp)
    mln.run(stim_duration)
    mln.set_input(0)
    mln.run(t_stop - (t_stim_start + stim_duration))
    mln.plot_history()

    # saving data
    data = dict()
    data['state'] = np.array(mln.state_history)
    data['t'] = np.array(mln.t_range)
    data['inp'] = np.array(mln.input_history)

    save_to = f"{root_folder}/data/runs/mln_data.pkl"
    pickle.dump(data, open(save_to, "wb+"))







