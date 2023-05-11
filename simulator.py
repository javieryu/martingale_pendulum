import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
import matplotlib.animation as animation

"""
Simulator related code goes in here.
"""


def simulate_pendulum(x_init, config):
    """
    Takes initial conditions and rolls out the dynamics for a fixed time frame.
    x_init: state at t = 0, [theta (radians), theta_dot (radians / s)]
    T: duration of simulation in seconds
    dt: discrete time-steps to simulate
    config: contains dynamics model parameters
    """
    m = config["m"]
    l = config["l"]
    b = config["b"]
    g = config["g"]
    T = config["T"]
    dt = config["dt"]

    n_steps = int(np.ceil(T / dt))

    state_traj = np.zeros((n_steps, 2))
    xy_traj = np.zeros_like(state_traj)
    state_traj[0, :] = x_init
    xy_traj[0, :] = get_xy(state_traj[0, 0], l)
    for step in range(1, n_steps):
        state_traj[step, :] = pendulum_dynamics(state_traj[step - 1, :], dt, m, l, b, g)
        xy_traj[step, :] = get_xy(state_traj[step, 0], l)

    return state_traj, xy_traj


def pendulum_continuous(x, m, l, b, g):
    """
    Describes the continuous-time dynamics of an unforced, damped pendulum.
    x = [theta, theta_dot]
    x = [0, 0] is upright; positive theta is clockwise.
    """
    theta, theta_dot = x
    theta_ddot = (m * g * l * np.sin(theta) - b * theta_dot) / (m * l**2)

    return np.array([theta_dot, theta_ddot])


def pendulum_dynamics(x, dt, m=1, l=1, b=1, g=9.81):
    """
    Describes the discrete-time dynamics of an unforced, damped pendulum.
    Integrates the continuous-time dynamics using RK4.
    """
    k1 = pendulum_continuous(x, m, l, b, g)
    k2 = pendulum_continuous(x + dt * 0.5 * k1, m, l, b, g)
    k3 = pendulum_continuous(x + dt * 0.5 * k2, m, l, b, g)
    k4 = pendulum_continuous(x + dt * k3, m, l, b, g)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def get_xy(theta, l):
    """Return the (x, y) coordinates of the bob at angle theta"""
    return l * np.sin(theta), l * np.cos(theta)


def animate(i, xy_traj, line, circle):
    """Update the animation at frame i."""
    x, y = xy_traj[i, :]
    line.set_data([0, x], [0, y])
    circle.set_center((x, y))


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

    # Set the plot limits so that the pendulum has room to swing!
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

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

    # Set the plot limits so that the pendulum has room to swing!
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

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


if __name__ == "__main__":
    ### Script ###
    # specify model params and init condition
    x_init_1 = np.array([0.1, 0])  # slightly right of upright
    x_init_2 = np.array([np.pi / 2, 0])  # 90 deg right
    config = {"m": 0.5, "l": 1.0, "b": 0.3, "g": 9.81, "dt": 0.05, "T": 10.0}

    # solve for trajectory
    state_traj_1, xy_traj_1 = simulate_pendulum(x_init_1, config)
    state_traj_2, xy_traj_2 = simulate_pendulum(x_init_2, config)

    fig = animate_two_traj(xy_traj_1, xy_traj_2)
    plt.show()
