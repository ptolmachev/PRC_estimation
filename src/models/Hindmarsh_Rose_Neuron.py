import json
import pickle
import numpy as np
from collections import deque
from copy import deepcopy
from matplotlib import pyplot as plt
from src.utils.utils import get_project_root

'''
 class Hindmarsh-Rose Neuron which computes time series, and accepts an input function
'''
class Hindmarsh_Rose_Neuron():
    def __init__(self, params):
        self.a = params["a"]
        self.I = params["I"]
        self.b = params["b"]
        self.r = params["r"]
        self.s = params["s"]
        self.x_R = params["x_R"]
        self.dt = params["dt"]
        self.noise_lvl = params["noise_lvl"]
        self.history_len = int(params["history_len"] / self.dt)
        self.N = 3
        self.state = np.ones((1,self.N))
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
        if state.shape[1] != self.N:
            raise ValueError("The second dimension of a state should be a number of internal variables")
        x = state[:, 0]
        y = state[:, 1]
        z = state[:, 2]
        rhs = np.zeros_like(state)
        rhs[:, 0] = y + self.a * x ** 2 - x ** 3 - z + self.I + self.input
        rhs[:, 1] = 1 - self.b * x ** 2 - y
        rhs[:, 2] = self.r * (self.s * (x - self.x_R) - z)
        return rhs.squeeze()

    def jac_rhs(self, state):
        # dF_1(x,y)/dx dF_1(x,y)/dy
        # dF_2(x,y)/dx dF_2(x,y)/dy
        x = state[:, 0]
        y = state[:, 1]
        z = state[:, 2]
        jac = np.zeros((self.N, self.N))
        jac[0, 0] = 2 * self.a * x - 3 * x ** 2
        jac[0, 1] = 1
        jac[0, 2] = -1
        jac[1, 0] = -2 * self.b * x
        jac[1, 1] = -1
        jac[1, 2] = 0
        jac[2, 0] = self.r * self.s
        jac[2, 1] = 0
        jac[2, 2] = -self.r
        return jac

    def step_(self):
        '''
        rk4 difference scheme
        '''
        k_s1 = self.dt * self.rhs_(self.state)
        k_s2 = self.dt * self.rhs_(self.state + k_s1 / 2)
        k_s3 = self.dt * self.rhs_(self.state + k_s2 / 2)
        k_s4 = self.dt * self.rhs_(self.state + k_s3)
        new_state = self.state + 1.0 / 6.0 * (k_s1 + 2 * k_s2 + 2 * k_s3 + k_s4) + self.noise_lvl * np.random.randn(1, self.N)
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
        state_array = np.array(self.state_history).squeeze()
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
        plt.savefig(f"{root_folder}/img/Hindmarsh_Rose_Neuron.png")
        plt.show(block=True)
        plt.close()
        return None


if __name__ == '__main__':
    model_name = 'Hindmarsh_Rose_Neuron'
    root_folder = get_project_root()
    params = json.load(open(f"{root_folder}/data/model_params/{model_name}_params.json", 'r'))
    dt = params["dt"]
    t_stop = params["t_stop"]
    stim_amp = 0.01
    t_stim_start = 5000
    stim_duration = 10 * dt
    hrn = Hindmarsh_Rose_Neuron(params)
    hrn.run(t_stim_start)
    hrn.set_input(stim_amp)
    hrn.run(stim_duration)
    hrn.set_input(0)
    hrn.run(t_stop - (t_stim_start + stim_duration))
    hrn.plot_history()

    # saving data
    root_folder = get_project_root()
    data = dict()
    data['state'] = np.array(hrn.state_history).squeeze()
    data['t'] = np.array(hrn.t_range).squeeze()
    data['inp'] = np.array(hrn.input_history).squeeze()

    save_to = f"{root_folder}/data/runs/hrn_data.pkl"
    pickle.dump(data, open(save_to, "wb+"))







