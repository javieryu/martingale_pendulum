import torch
import numpy as np


# Local Imports
import simulator
import train_model


def load_model(model_path):
    model = train_model.PendulumNetwork().to("cpu").to(torch.double)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model


true_dynamics = lambda state : simulator.pendulum_dynamics(state, 0.05, m=0.5, l=1, b=0.3, g=9.81)
learned_dynamics = lambda state: train_model.single_input_pred(model, state)


"""
Preliminary steps:
    1. load model
    2. define number of rollout steps
    3. define how many trajectories to evaluate
    4. define experiment name
"""

filename = "models/20230518-115451"
model = load_model(filename + "/model")
num_steps = 200 # T = 10, dt = 0.05
num_trajectories = 100
experiment_name = "no_shift"






"""
Run experiments:
    1. no shift: same conditions as during training
    2. covariate shift: mirror initial state conditions compared to training
    3. gradual concept shift: slowly changing pendulum length
    4. abrupt concept shift: the mass is now at the end of a double pendulum
"""
if experiment_name == "no_shift":
    init_bounds = ([0.0, 0.0], [np.pi / 2, 0.0])

    sim_config = {"m": 0.5, "l": 1.0, "b": 0.3, "g": 9.81, "dt": 0.05, "T": 10.0,
              "num_train_trajs": num_trajectories, "num_valid_trajs": 0,
              "train_init_bounds": init_bounds}
    
    trajectories = simulator.generate_trajectories(init_bounds, sim_config, num_trajectories)



elif experiment_name == "covariate_shift":
    None


elif experiment_name == "gradual_concept_shift":
    None


elif experiment_name == "abrupt_concept_shift":
    None


else:
    print("Invalid Experiment Name!")
    assert False














