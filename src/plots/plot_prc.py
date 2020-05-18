import json

import numpy as np
import pickle
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from src.models.Van_der_Pol_Oscillator import Van_der_Pol_Oscillator
from src.models.Morris_Lecar_Neuron import Morris_Lecar_Neuron
from src.models.Hodgkin_Huxley_Neuron import Hodgkin_Huxley_Neuron
from src.models.Hindmarsh_Rose_Neuron import Hindmarsh_Rose_Neuron
from src.utils.filtering_utils import scale
from src.utils.utils import get_project_root


def plot_prc(phi, delta_phi, model_name, stim_duration, stim_amp, signal = None, scatter=True, fit=False):
    fig = plt.figure(figsize=(20,10))
    tmp = list(zip(phi, delta_phi))
    tmp.sort(key = lambda a: a[0])
    phi, delta_phi = zip(*tmp)
    p = Polynomial.fit(phi, delta_phi, deg=30)
    if scatter == True:
        plt.scatter(np.array(phi), np.array(delta_phi), s=0.5, color='k', label="Integral Phase Change")
    else:
        plt.plot(np.array(phi), np.array(delta_phi), linewidth=3, color='k', label="Integral Phase Change")
    if fit == True:
        plt.plot((np.array(phi)), p(np.sort(np.array(phi))), ls = 'dashed', color='r', linewidth=2, label = "Polynomial Fit")
    if not signal is None:
        plt.plot((np.array(phi)), scale(signal), ls='-', color='b', linewidth=2, label="signal")
    plt.xlabel("Phase", fontsize=24)
    plt.ylabel("Delta Phi", fontsize=24)
    plt.legend(fontsize=24)
    plt.title(f"PRC {model_name}, stim_duration = {stim_duration}, stim_amp = {stim_amp}", fontdict={"size":24})
    plt.grid(True)
    return fig

if __name__ == '__main__':
    root_folder = get_project_root()

    #### plotting experimental PRC
    # model_name = 'Van_der_Pol_Oscillator'
    # model_name = 'Morris_Lecar_Neuron'
    # model_name = 'Hodgkin_Huxley_Neuron'
    # model_name = 'Hindmarsh_Rose_Neuron'
    # model_names = ['Van_der_Pol_Oscillator' , 'Morris_Lecar_Neuron', 'Hodgkin_Huxley_Neuron', 'Hindmarsh_Rose_Neuron']
    # model_names = ['Morris_Lecar_Neuron']
    # for model_name in model_names:
    #     modifier = "linear"
    #     params = json.load(open(f"{root_folder}/data/model_params/{model_name}_params.json", 'r'))
    #     dt = params["dt"]
    #     t_stop = params["t_stop"]
    #     max_amp = params["max_amp"]
    #     noise_lvl = 0
    #     amps = np.array([0.05, 0.2, 0.5, 1]) * max_amp
    #     stim_duration_multipliers = [1, 10, 100, 1000]
    #     for stim_duration_multiplier in stim_duration_multipliers:
    #         for stim_amp in amps:
    #             stim_duration = stim_duration_multiplier * dt
    #             tag = f"{dt}_{noise_lvl}_{stim_duration}_{stim_amp}"
    #             data_folder = f"{root_folder}/data/processed_data"
    #             data = pickle.load(open(f"{data_folder}/{model_name}_{modifier}_{tag}_prc.pkl", 'rb+'))
    #             phi = data["phi"]
    #             delta_phi = data["delta_phi"]
    #             fig = plot_prc(phi, delta_phi, model_name, stim_duration, stim_amp, signal = None, scatter=False, fit=False)
    #             plt.savefig(f"{root_folder}/img/{model_name}_{modifier}_{tag}.png")
    #             # plt.show(block=True)
    #             # plt.close()


    #### plotting exact PRC computed from full dynamics
    model_names = ['Hindmarsh_Rose_Neuron', 'Van_der_Pol_Oscillator', 'Morris_Lecar_Neuron', 'Hodgkin_Huxley_Neuron']
    # model_names = ['Hindmarsh_Rose_Neuron', 'Van_der_Pol_Oscillator', 'Morris_Lecar_Neuron']
    # model_names = ['Van_der_Pol_Oscillator']
    data_folder = f"{root_folder}/data/processed_data"
    for model_name in model_names:
        modifier = 'exact_fourier'
        tag = ''
        data = pickle.load(open(f"{data_folder}/{model_name}_{modifier}_{tag}_prc.pkl", 'rb+'))
        phi = np.array(data["phi"]) % (2 * np.pi)
        delta_phi = data["delta_phi"]
        signal_all = data["signal"]
        signal = data["signal"][:, 0]
        params = data["params"]
        omega = data["omega"]
        Z = data["Z"]
        Model = eval(f"{model_name}(params)")
        F = np.array([Model.rhs_(signal_all[i,:]) for i in range(signal_all.shape[0])])
        fig = plot_prc(phi, delta_phi, model_name, stim_duration=0, stim_amp=0, signal = signal, scatter=False, fit=False)
        plt.plot(phi, np.sum(Z * F, axis = 1) - omega, color = 'r')
        plt.savefig(f"{root_folder}/img/{model_name}_{modifier}.png")
        plt.show(block=True)
        plt.close()
