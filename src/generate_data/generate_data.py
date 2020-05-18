import json

from src.utils.phase_extraction import extract_protophase, extract_phase, get_period, get_period_
from src.utils.utils import get_project_root, create_dir_if_not_exist
from src.models.Van_der_Pol_Oscillator import Van_der_Pol_Oscillator
from src.models.Morris_Lecar_Neuron import Morris_Lecar_Neuron
from src.models.Hodgkin_Huxley_Neuron import Hodgkin_Huxley_Neuron
from src.models.Hindmarsh_Rose_Neuron import Hindmarsh_Rose_Neuron
import pickle
import numpy as np
from tqdm.auto import tqdm

def generate_data(M, params, model_name, stim_amp, stim_duration, t_stop, num_trials):
    root_folder = get_project_root()
    dt = params["dt"]
    noise_lvl = params["noise_lvl"]
    T_transient = params["T_transient"]
    num_transient_inds = int(T_transient/dt)
    M.run(t_stop)
    t = np.array(M.t_range).squeeze()
    signal = np.array(M.state_history).squeeze()[:, 0] ###
    protophase = extract_protophase(t, signal, -1, filter=False)
    phase = extract_phase(protophase, -1, n_bins=100, order=30)
    # disregard transients

    T = get_period_(t[num_transient_inds:], phase[num_transient_inds:])
    phase_wrapped = (phase) % (2 * np.pi)
    inds = np.where(np.diff(phase_wrapped[num_transient_inds:]) < -1.0 * np.pi)[0]
    initial_state = np.array(M.state_history)[num_transient_inds + inds[0] - 1, :].reshape(1, M.N) ###
    stim_start_baseline = (inds[7]) * dt  # start of the stimulation
    tag = f"{dt}_{noise_lvl}_{stim_duration}_{stim_amp}"
    dir_name = f"{model_name}_{tag}"
    save_to = f"{root_folder}/data/runs/{dir_name}"
    create_dir_if_not_exist(save_to)
    history_len = 15 * T / dt + 1
    M.history_len = int(np.ceil(history_len))
    t_stop = 15 * T
    for i in tqdm(range(num_trials)):
        M.reset()
        M.state = initial_state
        # run to discard transients
        t_stim_start = stim_start_baseline + (T * i / num_trials)
        M.run(t_stim_start)
        M.set_input(stim_amp)
        M.run(stim_duration)
        M.set_input(0)
        M.run(t_stop - (t_stim_start + stim_duration))
        # if (i * 100 / num_trials) % 10.0 == 0:
        #     M.plot_history()
        data = dict()
        data['params'] = params
        data['signal'] = np.array(M.state_history).squeeze()[:, 0] ###
        data['signal_all'] = np.array(M.state_history).squeeze()
        data['t'] = np.array(M.t_range).squeeze()
        data['inp'] = np.array(M.input_history).squeeze()
        file_name = f"{save_to}/{model_name}_data_{i}.pkl"
        pickle.dump(data, open(file_name, "wb+"))
    return None

if __name__ == '__main__':
    root_folder = get_project_root()
    # model_name = 'Van_der_Pol_Oscillator'
    # model_name = 'Morris_Lecar_Neuron'
    # model_name = 'Hodgkin_Huxley_Neuron'
    # model_name = 'Hindmarsh_Rose_Neuron'
    model_names = ['Van_der_Pol_Oscillator'] #, 'Morris_Lecar_Neuron', 'Hodgkin_Huxley_Neuron', 'Hindmarsh_Rose_Neuron'
    for model_name in model_names:
        params = json.load(open(f"{root_folder}/data/model_params/{model_name}_params.json", 'r'))
        dt = params["dt"]
        num_trials = 200
        t_stop = params["t_stop"]
        max_amp = params["max_amp"]
        amps = np.array([0.05, 0.2, 0.5, 1]) * max_amp
        stim_duration_multipliers = [1, 10, 100, 1000]
        for stim_amp in amps:
            for stim_duration_multiplier in stim_duration_multipliers:
                Model = eval(f"{model_name}(params)")
                stim_duration = stim_duration_multiplier*dt
                generate_data(Model, params, model_name, stim_amp, stim_duration, t_stop, num_trials)




