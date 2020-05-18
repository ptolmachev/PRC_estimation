import json
from src.utils.utils import get_project_root
root_folder = get_project_root()

model = "Van_der_Pol_Oscillator"
params = dict()
params["mu"] = 4
params["dt"] = 0.005
params["noise_lvl"] = 0
params["t_stop"] = 600
params["history_len"] = 100000
params["T_transient"] = 10
params["max_amp"] = 2
json.dump(params, open(f"{root_folder}/data/model_params/{model}_params.json", "w+"))

model = "Morris_Lecar_Neuron"
params = dict()
params["dt"] = 0.2
params["t_stop"] = 5000
params["history_len"] = 100000
params["noise_lvl"] = 0
params["T_transient"] = 100
params["max_amp"] = 40

params["C"] = 5.0
params["g_Ca"] = 4.0
params["V_Ca"] = 120
params["g_K"] = 8
params["V_K"] = -80
params["g_L"] = 2
params["V_L"] = -60
params["V1"] = -1.2
params["V2"] = 18
params["V3"] = 12
params["V4"] = 17.4
params["phi"] = (1.0 / 15.0)
params["I0"] = 40
json.dump(params, open(f"{root_folder}/data/model_params/{model}_params.json", "w+"))

model = "Hodgkin_Huxley_Neuron"
params = dict()
params["dt"] = 0.02
params["t_stop"] = 1000
params["history_len"] = 100000
params["noise_lvl"] = 0
params["T_transient"] = 200
params["max_amp"] = 35

params["C"] = 1
params["g_Na"] = 120
params["V_Na"] = 85.7
params["g_K"] = 36
params["V_K"] = -11
params["g_L"] = 0.3
params["V_L"] = 10.559
params["I0"] = 41
json.dump(params, open(f"{root_folder}/data/model_params/{model}_params.json", "w+"))

model = "Hindmarsh_Rose_Neuron"
params = dict()
params["dt"] = 0.1
params["t_stop"] = 10000
params["history_len"] = 100000
params["noise_lvl"] = 0
params["T_transient"] = 200
params["max_amp"] = 1.7

params["a"] = 3
params["I"] = 1.3
params["b"] = 5
params["r"] = 0.001
params["s"] = 4
params["x_R"] = -1.6
json.dump(params, open(f"{root_folder}/data/model_params/{model}_params.json", "w+"))