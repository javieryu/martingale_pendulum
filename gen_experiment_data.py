import torch
import numpy as np
import yaml
import pdb
import pickle
from copy import copy

# Local Imports
import simulator
import train_model

true_dynamics = lambda state: simulator.pendulum_dynamics(
    state, 0.05, m=0.5, l=1, b=0.3, g=9.81
)
learned_dynamics = lambda state: train_model.single_input_pred(model, state)


if __name__ == "__main__":
    """
    Preliminary steps:
        1. load model
        2. define number of rollout steps
        3. define how many trajectories to evaluate
        4. define experiment name
    """

    filename = "models/20230603-091943"
    model = train_model.load_model(filename + "/model")
    num_steps = 200  # T = 10, dt = 0.05
    num_trajectories = 1000
    experiment_name = "all"
    save_dir = "data"
    with open(filename + "/config.yaml") as file:
        train_config = yaml.full_load(file)

    train_config["num_valid_trajectories"] = 0

    """
    Run experiments:
        1. no shift: same conditions as during training
        2. covariate shift: mirror initial state conditions compared to training
        3. gradual concept shift: slowly changing pendulum length
        4. abrupt concept shift: the mass is now at the end of a double pendulum

    Data Formats:
        Each experiment type is saved to a pickle file in the data/ folder. Each pickle
        file is list of lists of trajectories.

        file = [[gt_states, pred_states, gt_xy, pred_xy, params], ...] with a length
            of num_trajectories.

        The params are the [m, l, b, g] of the simulation at each time step.
    """
    if experiment_name == "no_shift" or experiment_name == "all":
        """
        In distribution, generate trajectories and predict with exactly the
        same conditions as training.
        """
        init_bounds = ([np.pi / 2, 0.0], [np.pi, 0.0])
        exp_config = copy(train_config)
        states, xys, params = simulator.generate_trajectories(
            init_bounds, exp_config, num_trajectories
        )

        in_dist_data = []
        for k in range(num_trajectories):
            input_states = states[k][:-1, :]
            pred_states = train_model.trajectory_predict(model, input_states)
            gt_states = states[k][1:, :]
            gt_xy = xys[k][1:, :]
            ls = params[k][1:, 1]  # [m, l, b, g]
            pred_xy = simulator.get_xy(pred_states[:, 0], ls)
            in_dist_data.append([gt_states, pred_states, gt_xy, pred_xy, params[k]])

        with open(save_dir + "/no_shift.pkl", "wb") as f:
            pickle.dump(in_dist_data, f)

    if experiment_name == "covariate_shift" or experiment_name == "all":
        """
        Initialize the pendulum on a flipped domain. [1.5 pi, 3.99 pi] which
        is the opposite of where it was trained [0.0, 0.5 pi].
        """
        init_bounds = ([-np.pi / 2, 0.0], [-np.pi, 0.0])
        exp_config = copy(train_config)
        states, xys, params = simulator.generate_trajectories(
            init_bounds, exp_config, num_trajectories
        )

        cov_shift_data = []
        for k in range(num_trajectories):
            input_states = states[k][:-1, :]
            pred_states = train_model.trajectory_predict(model, input_states)
            gt_states = states[k][1:, :]
            gt_xy = xys[k][1:, :]
            ls = params[k][1:, 1]  # [m, l, b, g]
            pred_xy = simulator.get_xy(pred_states[:, 0], ls)
            cov_shift_data.append([gt_states, pred_states, gt_xy, pred_xy, params[k]])

        with open(save_dir + "/cov_shift.pkl", "wb") as f:
            pickle.dump(cov_shift_data, f)

    if experiment_name == "gradual_concept_shift" or experiment_name == "all":
        """
        Gradual concept shift, pendulum length changes linearly from 0.2 to 1.0 during the
        trajectory rollout.
        """
        init_bounds = ([np.pi / 2, 0.0], [np.pi, 0.0])
        exp_config = copy(train_config)
        exp_config["l"] = [1.0, 0.2]
        exp_config["m"] = [0.5, 1.0]
        states, xys, params = simulator.generate_trajectories(
            init_bounds, exp_config, num_trajectories
        )

        gcon_shift_data = []
        for k in range(num_trajectories):
            input_states = states[k][:-1, :]
            pred_states = train_model.trajectory_predict(model, input_states)
            gt_states = states[k][1:, :]
            gt_xy = xys[k][1:, :]
            ls = params[k][1:, 1]  # [m, l, b, g]
            pred_xy = simulator.get_xy(pred_states[:, 0], ls)
            gcon_shift_data.append([gt_states, pred_states, gt_xy, pred_xy, params[k]])

        with open(save_dir + "/gcon_shift.pkl", "wb") as f:
            pickle.dump(gcon_shift_data, f)

    if experiment_name == "abrupt_concept_shift" or experiment_name == "all":
        """
        Abrupt concept shift, double pendulum dynamics after training on single.
        """
        init_bounds = ([np.pi / 2, np.pi / 2, 0.0, 0.0], [np.pi, np.pi, 0.0, 0.0])
        exp_config = {
            "m1": [0.5, 0.5],
            "m2": [0.5, 0.5],
            "l1": [1.0, 1.0],
            "l2": [0.5, 0.5],
            "b1": [0.3, 0.3],
            "b2": [0.3, 0.3],
            "g": [9.81, 9.81],
            "dt": 0.05,
            "T": 10.0,
        }
        states, xys, params = simulator.generate_double_trajectories(
            init_bounds, exp_config, num_trajectories
        )

        acon_shift_data = []
        for k in range(num_trajectories):
            input_states = states[k][:-1, :]
            pred_states = train_model.trajectory_predict(model, input_states)
            gt_states = states[k][1:, :]
            gt_xy = xys[k][1:, :]
            l1s = params[k][1:, 2]  # [m, l, b, g]
            l2s = params[k][1:, 3]  # [m, l, b, g]
            pred_xy = simulator.get_double_xy(
                pred_states[:, 0], pred_states[:, 1], l1s, l2s
            )
            acon_shift_data.append([gt_states, pred_states, gt_xy, pred_xy, params[k]])

        with open(save_dir + "/acon_shift.pkl", "wb") as f:
            pickle.dump(acon_shift_data, f)
