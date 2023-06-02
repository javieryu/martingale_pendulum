import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
import matplotlib.animation as animation
import multiprocessing as mp
import itertools
import torch
import pdb

"""
Simulator related code goes in here.
"""


def generate_linear_mix_lambda(param):
    return lambda r: (1.0 - r) * param[0] + r * param[1]


def simulate_pendulum(x_init, config):
    """
    Takes initial conditions and rolls out the dynamics for a fixed time frame.
    x_init: state at t = 0, [theta (radians), theta_dot (radians / s)]
    T: duration of simulation in seconds
    dt: discrete time-steps to simulate
    config: contains dynamics model parameters
    """
    m = generate_linear_mix_lambda(config["m"])
    l = generate_linear_mix_lambda(config["l"])
    b = generate_linear_mix_lambda(config["b"])
    g = generate_linear_mix_lambda(config["g"])
    T = config["T"]
    dt = config["dt"]

    n_steps = int(np.ceil(T / dt))

    state_traj = np.zeros((n_steps, 2))
    param_traj = np.zeros((n_steps, 4))
    xy_traj = np.zeros_like(state_traj)
    state_traj[0, :] = x_init
    param_traj[0, :] = np.array([m(0), l(0), b(0), g(0)])
    xy_traj[0, :] = get_xy(state_traj[0, 0], l(0))
    for step in range(1, n_steps):
        r = step / n_steps
        state_traj[step, :] = pendulum_dynamics(
            state_traj[step - 1, :], dt, m(r), l(r), b(r), g(r)
        )
        xy_traj[step, :] = get_xy(state_traj[step, 0], l(r))
        param_traj[step, :] = np.array([m(r), l(r), b(r), g(r)])

    state_traj[:, 0] = np.mod(state_traj[:, 0], 2 * np.pi)
    return [state_traj, xy_traj, param_traj]


def simulate_double_pendulum(x_init, config):
    """
    Takes initial conditions and rolls out the dynamics for a fixed time frame.
    x_init: state at t = 0, [theta (radians), theta_dot (radians / s)]
    T: duration of simulation in seconds
    dt: discrete time-steps to simulate
    config: contains dynamics model parameters
    """
    m1 = generate_linear_mix_lambda(config["m1"])
    m2 = generate_linear_mix_lambda(config["m2"])
    l1 = generate_linear_mix_lambda(config["l1"])
    l2 = generate_linear_mix_lambda(config["l2"])
    b1 = generate_linear_mix_lambda(config["b1"])
    b2 = generate_linear_mix_lambda(config["b2"])
    g = generate_linear_mix_lambda(config["g"])
    T = config["T"]
    dt = config["dt"]

    n_steps = int(np.ceil(T / dt))

    state_traj = np.zeros((n_steps, 4))
    param_traj = np.zeros((n_steps, 7))
    xy_traj = np.zeros_like(state_traj)
    state_traj[0, :] = x_init
    param_traj[0, :] = np.array([m1(0), m2(0), l1(0), l2(0), b1(0), b2(0), g(0)])
    xy_traj[0, :] = get_double_xy(state_traj[0, 0], state_traj[0, 1], l1(0), l2(0))
    for step in range(1, n_steps):
        r = step / n_steps
        state_traj[step, :] = double_pendulum_dynamics(
            state_traj[step - 1, :], dt, m1(r), m2(r), l1(r), l2(r), b1(r), b2(r), g(r)
        )
        xy_traj[step, :] = get_double_xy(state_traj[step, 0], state_traj[step, 1], l1(r), l2(r))
        param_traj[step, :] = np.array([m1(r), m2(r), l1(r), l2(r), b1(r), b2(r), g(r)])

    state_traj[:, 0:2] = np.mod(state_traj[:, 0:2], 2 * np.pi)
    return [state_traj, xy_traj, param_traj]


def pendulum_continuous(x, m, l, b, g):
    """
    Describes the continuous-time dynamics of an unforced, damped pendulum.
    x = [theta, theta_dot]
    x = [0, 0] is upright; positive theta is clockwise.
    """
    theta, theta_dot = x
    theta_ddot = (m * g * l * np.sin(theta) - b * theta_dot) / (m * l ** 2)

    return np.array([theta_dot, theta_ddot])


def pendulum_dynamics(x, dt, m=1.0, l=1.0, b=1.0, g=9.81):
    """
    Describes the discrete-time dynamics of an unforced, damped pendulum.
    Integrates the continuous-time dynamics using RK4.
    """
    k1 = pendulum_continuous(x, m, l, b, g)
    k2 = pendulum_continuous(x + dt * 0.5 * k1, m, l, b, g)
    k3 = pendulum_continuous(x + dt * 0.5 * k2, m, l, b, g)
    k4 = pendulum_continuous(x + dt * k3, m, l, b, g)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def double_pendulum_continuous(x, m1, m2, l1, l2, b1, b2, g):
    """
    https://ir.canterbury.ac.nz/bitstream/handle/10092/12659/chen_2008_report.pdf
    """
    th1, th2, dth1, dth2 = x
    delth = th1 - th2
    alpha = b1 * dth1
    beta = b2 * dth2
    ddth1 = (
        m2 * l1 * (dth1 ** 2) * np.sin(2 * delth)
        + 2 * m2 * l2 * (dth2 ** 2) * np.sin(delth)
        + 2 * g * m2 * np.cos(th2) * np.sin(delth)
        + 2 * g * m1 * np.sin(th1)
        + 2 * alpha
        - 2 * beta * np.cos(delth)
    ) / (-2 * l1 * (m1 + m2 * np.sin(delth) ** 2))
    ddth2 = (
        m2 * l2 * (dth2 ** 2) * np.sin(2 * delth)
        + 2 * (m1 + m2) * l1 * (dth1 ** 2) * np.sin(delth)
        + 2 * g * (m1 + m2) * np.cos(th1) * np.sin(delth)
        + 2 * alpha * np.cos(delth)
        - 2 * (m1 + m2) * beta / m2
    ) / (2 * l2 * (m1 + m2 * np.sin(delth) ** 2))

    return np.array([dth1, dth2, ddth1, ddth2])


def double_pendulum_dynamics(x, dt, m1, m2, l1, l2, b1, b2, g):
    k1 = double_pendulum_continuous(x, m1, m2, l1, l2, b1, b2, g)
    k2 = double_pendulum_continuous(x + dt * 0.5 * k1, m1, m2, l1, l2, b1, b2, g)
    k3 = double_pendulum_continuous(x + dt * 0.5 * k2, m1, m2, l1, l2, b1, b2, g)
    k4 = double_pendulum_continuous(x + dt * k3, m1, m2, l1, l2, b1, b2, g)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_trajectories(
    init_bounds,
    sim_config,
    num_trajectories,
    init_dist="uniform",
    use_mp=True,
    workers=4,
):
    """
    Generates num_trajectories number of pendulum trajectories
    that are sampled from initial conditions with a specified
    support and distribution.

    init_bounds: state init bounds formatted as tuple(lows, highs) where
        lows/highs are lists of length = state_dim.
    sim_config: dictionary of configuration settings for simulator.
    num_trajectories: int, number of trajectories to simulate.
    init_dist: distribution over initial conditions.
        Options are ["uniform", ]
    use_mp: bool to use multiple threads to compute trajectories
    workers: int number of threads to use in multiprocessing

    Returns two lists of trajectories where each trajectory
    is a numpy array [time, state_dim], and the first list is theta state and
    the second list is xy state.
    """

    if init_dist == "uniform":
        state_inits = [np.random.uniform(*init_bounds) for _ in range(num_trajectories)]
    else:
        raise NameError("simulator.py: Unknown initial state distribution type.")

    if use_mp:
        pool = mp.Pool(workers)
        trajs = pool.starmap(
            simulate_pendulum, zip(state_inits, itertools.repeat(sim_config))
        )
    else:
        trajs = [simulate_pendulum(init, sim_config) for init in state_inits]

    return zip(*trajs)


def rad_to_cossin(trajectory):
    return torch.cat(
        (
            torch.cos(trajectory[:, 0]).unsqueeze(1),
            torch.sin(trajectory[:, 0]).unsqueeze(1),
            trajectory[:, 1:],
        ),
        dim=1,
    )


def cossin_to_rad(trajectory):
    return torch.cat(
        (
            torch.atan2(trajectory[:, 1], trajectory[:, 0]).unsqueeze(1),
            trajectory[:, 1:],
        ),
        dim=1,
    )


def get_xy(theta, l):
    """Return the (x, y) coordinates of the bob at angle theta"""
    return l * np.sin(theta), l * np.cos(theta)

def get_double_xy(theta1, theta2, l1, l2):
    """Return the (x, y) coordinates of the bobs of the double pendulums."""
    x1, y1 = l1 * np.sin(theta1), -l1 * np.cos(theta1)
    x2, y2 = x1 + l2 * np.sin(theta2), y1 - l2 * np.cos(theta2)
    return np.stack([x1, y1, x2, y2])

def animate(i, xy_traj, line, circle):
    """Update the animation at frame i."""
    x, y = xy_traj[i, :]
    line.set_data([0, x], [0, y])
    circle.set_center((x, y))

def animate_double(i, xy_traj, line, circle1, circle2):
    """ Update the animation for a sinlge frame of double pendulum."""
    x1, y1, x2, y2 = xy_traj[i, :]
    line.set_data([0, x1, x2], [0, y1, y2])
    circle1.set_center([x1, y1])
    circle2.set_center([x2, y2])

def animate_two(i, xy_traj_1, line_1, circle_1, xy_traj_2, line_2, circle_2):
    """Update the 2 traj animation at frame i."""
    x, y = xy_traj_1[i, :]
    line_1.set_data([0, x], [0, y])
    circle_1.set_center((x, y))

    x, y = xy_traj_2[i, :]
    line_2.set_data([0, x], [0, y])
    circle_2.set_center((x, y))


def animate_single_traj(xy_traj):
    """Makes an animation of a single trajectory"""
    fig = plt.figure()
    ax = fig.add_subplot(aspect="equal")

    # The pendulum rod, in its initial position.
    (line,) = ax.plot([0, xy_traj[0, 0]], [0, xy_traj[0, 1]], lw=3, c="black")

    # The pendulum bob: set zorder so that it is drawn over the pendulum rod.
    bob_radius = 0.08
    circle = ax.add_patch(plt.Circle(xy_traj[0, :], bob_radius, fc="red", zorder=3))

    # Set the plot limits so that the pendulum has room to swing
    l = np.sqrt(xy_traj[0, 0] ** 2 + xy_traj[0, 1] ** 2)  # lenth of pendulum
    ax.set_xlim(-1.2 * l, 1.2 * l)
    ax.set_ylim(-1.2 * l, 1.2 * l)

    nframes = xy_traj.shape[0]
    ani = animation.FuncAnimation(
        fig,
        lambda i: animate(i, xy_traj, line, circle),
        frames=range(nframes),
        repeat=True,
        interval=0.05 * 1000,
    )
    fig.ani = ani
    return fig

def animate_double_traj(xy_traj):
    """Makes an animation of a double pendulum trajectory"""
    fig = plt.figure()
    ax = fig.add_subplot(aspect="equal")

    # The pendulum rod, in its initial position.
    (line,) = ax.plot([0, xy_traj[0, 0], xy_traj[0, 2]], [0, xy_traj[0, 1], xy_traj[0, 3]], lw=3, c="black")

    # The pendulum bob: set zorder so that it is drawn over the pendulum rod.
    bob_radius = 0.08
    circle1 = ax.add_patch(plt.Circle(xy_traj[0, :2], bob_radius, fc="red", zorder=3))
    circle2 = ax.add_patch(plt.Circle(xy_traj[0, 2:], bob_radius, fc="red", zorder=3))

    # Set the plot limits so that the pendulum has room to swing
    l = np.sqrt(xy_traj[0, 2] ** 2 + xy_traj[0, 3] ** 2)  # lenth of pendulum
    ax.set_xlim(-1.2 * l, 1.2 * l)
    ax.set_ylim(-1.2 * l, 1.2 * l)

    nframes = xy_traj.shape[0]
    ani = animation.FuncAnimation(
        fig,
        lambda i: animate_double(i, xy_traj, line, circle1, circle2),
        frames=range(nframes),
        repeat=True,
        interval=0.05 * 1000,
    )
    fig.ani = ani
    return fig

def animate_two_traj(xy_traj_1, xy_traj_2):
    """Makes an animation of two trajectories with xy_traj_2 transparent"""
    fig = plt.figure()
    ax = fig.add_subplot(aspect="equal")

    # The pendulum rod, in its initial position.
    (line_1,) = ax.plot([0, xy_traj_1[0, 0]], [0, xy_traj_1[0, 1]], lw=3, c="black")
    (line_2,) = ax.plot(
        [0, xy_traj_2[0, 0]], [0, xy_traj_2[0, 1]], lw=3, c="black", alpha=0.5
    )

    # The pendulum bob: set zorder so that it is drawn over the pendulum rod.
    bob_radius = 0.08
    circle_1 = ax.add_patch(plt.Circle(xy_traj_1[0, :], bob_radius, fc="red", zorder=3))
    circle_2 = ax.add_patch(
        plt.Circle(xy_traj_2[0, :], bob_radius, fc="red", zorder=3, alpha=0.5)
    )

    # Set the plot limits so that the pendulum has room to swing
    l = max(
        np.sqrt(xy_traj_1[0, 0] ** 2 + xy_traj_1[0, 1] ** 2),
        np.sqrt(xy_traj_2[0, 0] ** 2 + xy_traj_2[0, 1] ** 2),
    )
    ax.set_xlim(-1.2 * l, 1.2 * l)
    ax.set_ylim(-1.2 * l, 1.2 * l)

    nframes = min(xy_traj_1.shape[0], xy_traj_2.shape[0])
    ani = animation.FuncAnimation(
        fig,
        lambda i: animate_two(
            i, xy_traj_1, line_1, circle_1, xy_traj_2, line_2, circle_2
        ),
        frames=range(nframes),
        repeat=True,
        interval=0.05 * 1000,
    )
    fig.ani = ani
    return fig


def make_pendulum_figure(theta):
    fig = plt.figure()
    ax = fig.add_subplot(aspect="equal")

    x_bob = np.sin(theta)
    y_bob = np.cos(theta)
    # The pendulum rod, in its initial position.
    (line,) = ax.plot([0, x_bob], [0, y_bob], lw=3, c="black")

    # The pendulum bob: set zorder so that it is drawn over the pendulum rod.
    bob_radius = 0.08
    circle = ax.add_patch(plt.Circle([x_bob, y_bob], bob_radius, fc="red", zorder=3))

    # Set the plot limits so that the pendulum has room to swing
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # add grid lines
    plt.grid()

    # add arc
    arc_angles = np.linspace(0, theta, 20)
    arc_xs = 0.7 * np.sin(arc_angles)
    arc_ys = 0.7 * np.cos(arc_angles)
    plt.plot(arc_xs, arc_ys, color="black", lw=1)

    # add arrow to arc
    plt.arrow(
        arc_xs[-3],
        arc_ys[-3],
        0.01,
        -0.005,
        color="black",
        width=0.001,
        head_width=0.02,
        head_length=0.02,
    )

    # add theta label
    plt.text(0.17, 0.75, r"$\theta$", fontsize=15)

    plt.savefig("figures/pend_fig.svg")
    plt.savefig("figures/pend_fig.png")


if __name__ == "__main__":
    ### Script Single ###
    # specify model params and init condition
    # x_init_1 = np.array([0.1, 0])  # slightly right of upright
    # x_init_2 = np.array([3.9 * np.pi / 2, 0])  # 90 deg right
    # config1 = {
    #     "m": [0.5, 0.5],
    #     "l": [1.0, 1.0],
    #     "b": [0.3, 0.3],
    #     "g": [9.81, 9.81],
    #     "dt": 0.05,
    #     "T": 10.0,
    # }
    # config2 = {
    #     "m": [0.5, 0.5],
    #     "l": [1.0, 0.5],
    #     "b": [0.3, 0.3],
    #     "g": [9.81, 9.81],
    #     "dt": 0.05,
    #     "T": 10.0,
    # }
    #
    # # make and save pendulum figure
    # # make_pendulum_figure(np.pi/6)
    #
    # # solve for trajectory
    # state_traj_1, xy_traj_1, _ = simulate_pendulum(x_init_1, config1)
    # state_traj_2, xy_traj_2, _ = simulate_pendulum(x_init_2, config2)
    #
    # fig = animate_two_traj(xy_traj_1, xy_traj_2)
    # plt.show()

    ### Script double ###
    x_init_1 = np.array([0.9 * np.pi, 0.9 * np.pi, 0.0, 0.0])  # slightly right of upright
    config1 = {
        "m1": [0.5, 0.5],
        "m2": [0.5, 0.5],
        "l1": [1.0, 1.0],
        "l2": [1.0, 1.0],
        "b1": [0.3, 0.3],
        "b2": [0.3, 0.3],
        "g": [9.81, 9.81],
        "dt": 0.05,
        "T": 10.0,
    }
    state_traj_1, xy_traj_1, _ = simulate_double_pendulum(x_init_1, config1)
    fig = animate_double_traj(xy_traj_1)
    plt.show()
