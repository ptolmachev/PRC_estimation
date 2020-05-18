import json
import pickle
from copy import deepcopy
import numpy as np
from tqdm.auto import tqdm
# original phase is omega * t + c
from src.utils.phase_extraction import extract_protophase, extract_phase, get_period_
from src.utils.utils import get_project_root, get_files


def get_phase_shifts(t, phase, inp):
    dt = t[2] - t[1]
    stim_start_ind = int(np.where(np.diff(inp) > 0)[0][0]) + 1
    stim_duration_ind = (int(np.where(np.diff(inp) < 0)[0][0]) - int(np.where(np.diff(inp) > 0)[0][0]))
    amp = inp[stim_start_ind]
    T = get_period_(t, phase)
    omega = 2 * np.pi / T
    Phis = np.array(phase[stim_start_ind: stim_start_ind + stim_duration_ind + 1])
    dphi_dt = np.diff(Phis) / dt
    Delta_Phis = (dphi_dt - omega) / inp[stim_start_ind: stim_start_ind + stim_duration_ind]
    return Phis % (2 * np.pi), Delta_Phis

def run_prc_extraction(files, save_to):
    data_phi = []
    data_delta_phi = []
    for file in tqdm(files):
        data = pickle.load(open(f"{data_folder}/{file}", "rb+"))
        signal = data["signal"]
        t = data["t"]
        dt = t[2] - t[1]
        inp = data["inp"]
        stim_start_ind = int(np.where(np.diff(inp) > 0)[0][0])
        stim_duration_ind = (int(np.where(np.diff(inp) < 0)[0][0]) - int(np.where(np.diff(inp) > 0)[0][0]))
        protophase = extract_protophase(t, signal, stim_start_ind, filter=False)
        phase = extract_phase(protophase, stim_start_ind, n_bins=100, order=30)
        phis, delta_phis = get_phase_shifts(t, phase, inp)
        data_phi.extend(deepcopy(phis))
        data_delta_phi.extend(deepcopy(delta_phis))

    prc_data = dict()
    prc_data['phi'] = data_phi
    prc_data['delta_phi'] = data_delta_phi
    root_folder = get_project_root()
    pickle.dump(prc_data, open(f"{root_folder}/data/processed_data/{save_to}", "wb+"))
    return None

if __name__ == '__main__':
    model_names = ['Van_der_Pol_Oscillator','Morris_Lecar_Neuron', 'Hodgkin_Huxley_Neuron', 'Hindmarsh_Rose_Neuron']
    for model_name in model_names:
        modifer = "regression"
        root_folder = get_project_root()
        params = json.load(open(f"{root_folder}/data/model_params/{model_name}_params.json", 'r'))
        dt = params["dt"]
        t_stop = params["t_stop"]
        noise_lvl = params["noise_lvl"]
        max_amp = params["max_amp"]
        amps = np.array([0.05, 0.2, 0.5, 1]) * max_amp
        stim_duration_multipliers = [1, 10, 100, 1000]
        counter = 0
        for stim_amp in amps:
            for stim_duration_multiplier in stim_duration_multipliers:
                stim_duration = stim_duration_multiplier * dt
                root_folder = get_project_root()
                tag = f"{dt}_{noise_lvl}_{stim_duration}_{stim_amp}"
                dir_name = f"{model_name}_{tag}"
                data_folder = f"{root_folder}/data/runs/{dir_name}"
                files = get_files(data_folder, pattern=f"{model_name}_data_[0-9]*.pkl")
                save_to = f"{model_name}_{modifer}_{tag}_prc.pkl"
                run_prc_extraction(files, save_to)