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
import math

# Local Imports
import simulator

"""
Dynamics approximation code goes in here.
"""


class DoublePendulumNetwork(nn.Module):
    """
    Implements a basic feed forward neural network that uses
    ReLU activations for all of the layers.
    """

    def __init__(
        self,
    ):
        """Constructor for network.

        Args:
            shape (list of ints): list of network layer shapes, which
            includes the input and output layers.
        """
        super(DoublePendulumNetwork, self).__init__()

        # Build up the layers
        layers = []
        # (3, 64, 64, 64, 3)
        layers.append(nn.Linear(6, 64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(64, 64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(64, 64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(64, 6))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass on the input through the network.

        Args:
            x (torch.Tensor): Input tensor dims [batch, self.shape[0]]

        Returns:
            torch.Tensor: Output of network. [batch, self.shape[-1]]
        """
        outs = self.seq(x)
        normalized_cossin = F.normalize(outs[:, :-1])
        angular_vel = outs[:, -1].unsqueeze(1)

        return torch.cat((normalized_cossin, angular_vel), dim=-1)


def double_pendulum_loss(prediction, actual):
    angle_loss = -F.cosine_similarity(prediction[:, :-2], actual[:, :-2])
    vel_loss = F.mse_loss(prediction[:, -2:], actual[:, -2:])
    return angle_loss.mean() + vel_loss.mean()


def train_model(train_data, valid_data, use_cuda=True):
    """
    Trains a model on a bunch of trajectories.
    train_data: tuple(input_tensor, output_tensor) [N, in_dim] x [N, out_dim]
                everything a pytorch tensor.
    valid_data: tuple(input_tensor, output_tensor) [N, in_dim] x [N, out_dim]
                everything a pytorch tensor.
    use_cuda: bool to use cuda for everything
    """
    BATCH_SIZE = 2048
    NUM_EPOCHS = 10
    LR = 0.0001

    if torch.cuda.is_available() and use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = DoublePendulumNetwork().to(device).to(torch.double)

    # CONVERT DATA TO TENSORS

    train_data = map(lambda x: x.to(device), train_data)
    valid_data = map(lambda x: x.to(device), valid_data)

    train_dataset = TensorDataset(*train_data)
    valid_dataset = TensorDataset(*valid_data)

    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loss_table = Table()
    loss_table.add_column("Progress")
    loss_table.add_column("Avg. Train Loss")
    loss_table.add_column("Valid Loss")
    with Live(loss_table, refresh_per_second=4):
        for e in range(NUM_EPOCHS):
            with torch.no_grad():
                valid_losses = []
                for batch_inputs, batch_outputs in validloader:
                    pred_outputs = model.forward(batch_inputs)
                    valid_loss = double_pendulum_loss(pred_outputs, batch_outputs)
                    valid_losses.append(valid_loss.mean().to("cpu"))

            train_losses = []
            for batch_inputs, batch_outputs in trainloader:
                pred_outputs = model.forward(batch_inputs)
                train_loss = double_pendulum_loss(pred_outputs, batch_outputs)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                train_losses.append(train_loss.mean().to("cpu"))

            progress_percent = (e + 1) * 100 / NUM_EPOCHS
            mean_valid = sum(valid_losses) / len(valid_losses)
            mean_train = sum(train_losses) / len(train_losses)
            loss_table.add_row(
                f"{progress_percent:.1f}%", f"{mean_train:.3f}", f"{mean_valid:.3f}"
            )

    return model


def train_predictor(
    train_init_bounds,
    sim_config,
    num_valid_trajs=100,
    valid_init_bounds=None,
):
    """
    Train a predictor on a set of simulated trajectories.
    init_bounds: state init bounds formatted as tuple(lows, highs) where
        lows/highs are lists of length = state_dim.
    sim_config: dictionary of configuration settings for simulator.

    """
    num_train_trajs = sim_config["num_train_trajs"]
    # Generate Train Data
    status = f"Generating {num_train_trajs} train trajectories."
    with Status(status):
        train_trajs, _, _ = simulator.generate_double_trajectories(
            train_init_bounds, sim_config, num_train_trajs
        )

    # Generate valid Data
    if valid_init_bounds is None:
        valid_init_bounds = train_init_bounds

    status = f"Generating {num_valid_trajs} validation trajectories"
    with Status(status):
        valid_trajs, _, _ = simulator.generate_double_trajectories(
            valid_init_bounds, sim_config, num_valid_trajs
        )

    # Stack and convert to torch
    train_trajs = torch.from_numpy(np.concatenate(train_trajs))
    valid_trajs = torch.from_numpy(np.concatenate(valid_trajs))

    # Convert to cos-sin representation
    train_trajs = simulator.double_rad_to_cossin(train_trajs)
    valid_trajs = simulator.double_rad_to_cossin(valid_trajs)

    # Split Data
    train_data = (train_trajs[:-1], train_trajs[1:])
    valid_data = (valid_trajs[:-1], valid_trajs[1:])

    model = train_model(train_data, valid_data, use_cuda=True)
    return model


def rollout_predictor(model, inputs, output_device="cpu"):
    inputs = inputs.to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model.forward(inputs)
    return outputs.to(torch.device(output_device))


def single_input_pred(model, state):
    state = np.array([[np.cos(state[0]), np.sin(state[0]), state[1]]])
    state = torch.from_numpy(state)
    pred = rollout_predictor(model, state).numpy()[0]
    pred = [math.atan2(pred[1], pred[0]), pred[2]]
    return pred

def trajectory_predict(model, states):
    nn_states = np.array([np.cos(states[:, 0]), np.sin(states[:, 0]), states[:, 1]])
    nn_states = torch.from_numpy(nn_states).T
    pred = rollout_predictor(model, nn_states).numpy()
    pred = simulator.double_cossin_to_rad(pred)
    return pred

def save_model(model, config):
    # make folder for training run
    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filepath = "models_double/" + str(current_datetime)
    os.mkdir(filepath)

    # save network and config dict
    torch.save(model.state_dict(), filepath + "/model")
    with open(filepath + "/config.yaml", "w") as file:
        yaml.dump(config, file)


def load_model(model_path):
    model = DoublePendulumNetwork().to("cpu").to(torch.double)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model


if __name__ == "__main__":
    train_init_bounds = ([np.pi / 2, np.pi / 2, 0.0, 0.0], [np.pi, np.pi, 0.0, 0.0])
    num_train_trajs = 8000
    num_valid_trajs = 200
    train_config = {
        "m1": [1.0, 1.0],
        "m2": [1.0, 1.0],
        "l1": [1.0, 1.0],
        "l2": [1.0, 1.0],
        "b1": [0.3, 0.3],
        "b2": [0.3, 0.3],
        "g": [9.81, 9.81],
        "dt": 0.05,
        "T": 10.0,
        "num_train_trajs": num_train_trajs,
        "num_valid_trajs": num_valid_trajs,
        "train_init_bounds": train_init_bounds,
    }
    model = train_predictor(train_init_bounds, train_config)
    save_model(model, train_config)
