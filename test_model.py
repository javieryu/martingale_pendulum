import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from rich.table import Table
from rich.live import Live
from rich.status import Status
import matplotlib.pyplot as plt
import pdb
import time
import yaml
from datetime import datetime
import os


# Local Imports
import simulator
import train_model


def load_model(model_path):
    model = train_model.PendulumNetwork().to("cpu").to(torch.double)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model


def training_dist_plot(model_path):
    with open(model_path + "/config.yaml") as file:
        config = yaml.full_load(file)

    train_init_bounds = config["train_init_bounds"]
    num_train_trajs = 100

    train_trajs, _ = simulator.generate_trajectories(train_init_bounds, config, num_train_trajs)

    fig = plt.figure()
    ax = fig.add_subplot(aspect="equal")
    plt.xlabel(r'$\theta$ (rad)', fontsize=15)
    plt.ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=15)
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2*np.pi, 2*np.pi)
    plt.grid()

    for traj in train_trajs:
        plt.scatter(traj[:,0], traj[:,1], color = "lightskyblue", s=10, alpha=0.5)

    
    plt.savefig("figures/training_dist.svg")
    plt.savefig("figures/training_dist.png")
    
    plt.show()


def quiver_plots(forward_dynamics):
    # solve for "phase diagram"
    thetas = np.linspace(-2*np.pi, 2*np.pi, num=50)
    theta_dots = np.linspace(-2*np.pi, 2*np.pi, num=50)
    Theta, Theta_dot = np.meshgrid(thetas, theta_dots)
    theta_dir = np.zeros(Theta.shape)
    theta_dot_dir = np.zeros(Theta.shape)
    for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            theta = Theta[i,j]
            theta_dot = Theta_dot[i,j]
            next_state = forward_dynamics(np.array([theta, theta_dot]))
            theta_dir[i,j] = next_state[0] - theta
            theta_dot_dir[i,j] = next_state[1] - theta_dot

    # make figure
    fig = plt.figure()
    ax = fig.add_subplot(aspect="equal")
    ax.quiver(Theta, Theta_dot, theta_dir, theta_dot_dir)
    plt.xlabel(r'$\theta$ (rad)', fontsize=15)
    plt.ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=15)
    plt.show()




if __name__ == "__main__":
    filename = "models/20230518-115451"
    model = load_model(filename + "/model")
    test_config = {"m": 0.5, "l": 1.0, "b": 0.3, "g": 9.81, "dt": 0.05, "T": 10.0}
    init_state = [-np.pi/6, 0]
    
    # compute rollouts
    # test_traj, test_xy = simulator.simulate_pendulum(np.array(init_state), test_config)
    # test_inputs = simulator.rad_to_cossin(torch.from_numpy(test_traj[:-1]))
    # test_preds = train_model.rollout_predictor(model, test_inputs)
    # test_preds_xy = test_config["l"] * test_preds[:, :-1].flip(-1)

    # # animate rollouts
    # fig = simulator.animate_two_traj(test_xy[1:], test_preds_xy)
    # plt.show()


    # training_dist_plot(filename)
    true_dynamics = lambda state : simulator.pendulum_dynamics(state, 0.05, m=0.5, l=1, b=0.3, g=9.81)
    learned_dynamics = lambda state: train_model.single_input_pred(model, state)
    
    quiver_plots(true_dynamics)
    quiver_plots(learned_dynamics)
