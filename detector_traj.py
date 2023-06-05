import numpy as np
from confseq import conjmix_bounded as conjmix
import pickle
import matplotlib.pyplot as plt
import os
import pdb

"""
In this file we determine whether a given set of initial conditions will lead to harmful distribution shift or not.

This file has code for
    1. upper bounding the source risk
    2. lower bounding the target risk

The PM-H, PM-EB, and Betting confidence sequences assume that target data consists of iid samples from an unknown distribution
The Betting bound is currently not implemented
"""


def source_risk_ub(pred_seqs, true_seqs, delta):
    """
    Use PM-EB method to upper bound source risk
    Risk is the expected value of absolute error
    pred_seq: array of network predictions where pred_seq.shape = (# predictions, prediction dimension)
    true_seq: array of true labels corresponding to each network prediction
    delta: upper bound holds with probability 1-delta
    """
    risks = [
        np.linalg.norm(pred_seqs[i] - true_seqs[i], ord=np.inf)
        for i in range(len(pred_seqs))
    ]  # risk = max(sum(abs(pred_seq - true_seq), axis=1))

    _, ubs = bounds_pm_eb(risks, delta)

    ub_source = ubs[-1]

    return ub_source


def sequential_test(pred_seqs, true_seqs, delta, tol, ub_source, method="PM-H"):
    """
    pred_seq: array of network predictions where pred_seq.shape = (# predictions, prediction dimension)
    true_seq: array of true labels corresponding to each network prediction
    delta: target risk lower bounds holds with probability 1-delta
    tol: how much slack between source and target risk to allow before raising alert
    ub_source: upper bound on the source risk
    Risk is expected value of absolute error
    """

    risks = [
        np.linalg.norm(pred_seqs[i] - true_seqs[i], ord=np.inf)
        for i in range(len(pred_seqs))
    ]  # risk = max(sum(abs(pred_seq - true_seq), axis=1))

    if method == "PM-H":
        lbs_target, _ = bounds_pm_h(risks, delta)
    elif method == "PM-EB":
        lbs_target, _ = bounds_pm_eb(risks, delta)
    elif method == "Betting":
        lbs_target = bounds_betting(risks, delta)
    elif method == "CM-EB":
        lbs_target = bounds_cm_eb(risks, delta)

    alerts = lbs_target > (ub_source + tol)  # make boolean array of if alert is raised

    return lbs_target, alerts, np.array(risks)


def bounds_pm_h(Zs, delta):
    """
    Predictably-mixed Hoeffding's confidence sequence
    """
    T = len(Zs)
    lb = np.zeros(T)
    ub = np.zeros(T)
    lambdas = np.zeros(T)
    psis = np.zeros(T)
    for idx, t in enumerate(range(1, T + 1)):
        lambdas[idx] = min(1, np.sqrt((8 * np.log(1 / delta)) / (t * np.log(t + 1))))
        psis[idx] = lambdas[t] ** 2 / 8

        # compute first term in lb/ub
        term1 = np.dot(lambdas, Zs) / np.sum(lambdas)

        # compute second term in lb/ub
        term2 = (np.log(1 / delta) + np.sum(psis)) / np.sum(lambdas)

        # compute lb
        lb[idx] = term1 - term2
        ub[idx] = term1 + term2

    return lb, ub


def bounds_pm_eb(Zs, delta):
    """
    Predictably-mixed empirical-Bernstein confidence sequence
    """
    T = len(Zs)
    lbs = np.zeros(T)
    ubs = np.zeros(T)
    lambdas = np.zeros(T)
    psis = np.zeros(T)
    vs = np.zeros(T)
    mus = np.zeros(T)
    c = 0.5
    for idx, t in enumerate(range(1, T + 1)):
        mus[idx] = (0.5 + np.sum(Zs[: idx + 1])) / (t + 1)
        var = (0.25 + np.sum(((Zs - mus) ** 2)[: idx + 1])) / (t + 1)
        lambdas[idx] = min(
            c, np.sqrt((2 * np.log(1 / delta)) / (var * t * np.log(t + 1)))
        )

        if t == 1:
            vs[idx] = 4 * (Zs[idx]) ** 2  # assume initial empirical mean is zero
        else:
            vs[idx] = 4 * (Zs[idx] - mus[idx - 1]) ** 2
        psis[idx] = (-np.log(1 - lambdas[idx]) - lambdas[idx]) / 4

        # compute first term in lb/ub
        term1 = np.dot(lambdas, Zs) / np.sum(lambdas)

        # compute second term in lb/ub
        term2 = (np.log(1 / delta) + np.dot(vs, psis)) / np.sum(lambdas)

        # compute lb
        lbs[idx] = term1 - term2
        ubs[idx] = term1 + term2

    return lbs, ubs


def bounds_betting(Z):
    """
    Betting-based confidence sequence.
    Might not implement.
    """
    None
    return None


def bounds_cm_eb(Zs, delta):
    """
    Conjugate-Mixture Empirical-Bernstein confidence sequence
    """
    v_opt = len(Zs) / 2  # make bound tightest halfway through traj
    lbs = conjmix.conjmix_empbern_lower_cs(
        Zs, v_opt, alpha=delta, running_intersection=False
    )
    return lbs


def find_first_alert(seq):
    ind = None
    flag = True
    for t in range(seq.shape[0]):
        if seq[t] > 0:
            ind = t
            flag = False
            break

    if flag:
        ind = -1

    return ind


if __name__ == "__main__":
    # define hyperparameters
    delta = 0.1
    tol = 1e-4
    lb_method = "PM-EB"

    # load the in-distribution test data
    # data format: list of lists. Outer list is each traj. Inner lists are gt_states, pred_states, gt_xy, pred_xy, params
    file = open("data/no_shift.pkl", "rb")
    data_no_shift = pickle.load(file)
    file.close()

    file = open("data/cov_shift.pkl", "rb")
    data_cov_shift = pickle.load(file)
    file.close()

    file = open("data/gcon_shift.pkl", "rb")
    data_gcon_shift = pickle.load(file)
    file.close()

    file = open("data/acon_shift.pkl", "rb")
    data_acon_shift = pickle.load(file)
    file.close()

    num_trajs = len(data_no_shift)
    T = len(data_no_shift[0][0])
    ts = np.arange(1, T + 1, 1)
    print("Number of trajectories: ", num_trajs)
    print("Length of each trajectory: ", T)

    # compute upper bound on source risk
    pred_seqs_id = [data_no_shift[i][1] for i in range(num_trajs)]
    true_seqs_id = [data_no_shift[i][0] for i in range(num_trajs)]
    ub_source = source_risk_ub(pred_seqs_id, true_seqs_id, delta)
    print("Source Risk UB: ", ub_source)

    # set up plots
    # no shift
    plt.figure(0)
    plt.axhline(y=ub_source, color="black", linestyle="-")
    plt.xlabel(r"Sample Size, $n$")
    plt.ylabel(r"Risk, $\mathbb{E}[\Vert \hat{x}_{t+1} - x_{t+1} \Vert_1]$")

    # covariate shift
    plt.figure(1)
    plt.axhline(y=ub_source, color="black", linestyle="-")
    plt.xlabel(r"Sample Size, $n$")
    plt.ylabel(r"Risk, $\mathbb{E}[\Vert \hat{x}_{t+1} - x_{t+1} \Vert_1]$")

    # gradual concept shift
    plt.figure(2)
    plt.axhline(y=ub_source, color="black", linestyle="-")
    plt.xlabel(r"Sample Size, $n$")
    plt.ylabel(r"Risk, $\mathbb{E}[\Vert \hat{x}_{t+1} - x_{t+1} \Vert_1]$")

    # abrupt concept shift
    plt.figure(3)
    plt.axhline(y=ub_source, color="black", linestyle="-")
    plt.xlabel(r"Sample Size, $n$")
    plt.ylabel(r"Risk, $\mathbb{E}[\Vert \hat{x}_{t+1} - x_{t+1} \Vert_1]$")

    # set up and run experiment
    num_detections = {"no_shift": 0, "cov_shift": 0, "gcon_shift": 0, "acon_shift": 0}
    num_tests = 35
    sample_size = np.floor(1000 / num_tests)
    ns = np.arange(1, sample_size + 1, 1)
    for k in range(num_tests):
        print("On test ", k, "/", num_tests)

        # get predicted and true sequences for this traj
        pred_seqs_no_shift = [
            data_no_shift[i][1]
            for i in range(int(sample_size * k), int(sample_size * k + sample_size))
        ]
        true_seqs_no_shift = [
            data_no_shift[i][0]
            for i in range(int(sample_size * k), int(sample_size * k + sample_size))
        ]

        pred_seqs_cov_shift = [
            data_cov_shift[i][1]
            for i in range(int(sample_size * k), int(sample_size * k + sample_size))
        ]
        true_seqs_cov_shift = [
            data_cov_shift[i][0]
            for i in range(int(sample_size * k), int(sample_size * k + sample_size))
        ]

        pred_seqs_gcon_shift = [
            data_gcon_shift[i][1]
            for i in range(int(sample_size * k), int(sample_size * k + sample_size))
        ]
        true_seqs_gcon_shift = [
            data_gcon_shift[i][0]
            for i in range(int(sample_size * k), int(sample_size * k + sample_size))
        ]

        idxs = [0, 2]
        # print(data_acon_shift[0][1].shape)
        # print(data_acon_shift[0][0].shape)
        # print(data_acon_shift[0][0][:,idxs])
        # assert False
        pred_seqs_acon_shift = [
            data_acon_shift[i][1]
            for i in range(int(sample_size * k), int(sample_size * k + sample_size))
        ]
        true_seqs_acon_shift = [
            data_acon_shift[i][0][:, idxs]
            for i in range(int(sample_size * k), int(sample_size * k + sample_size))
        ]

        # compute lower bound on target risk
        # no shift
        lb_no_shift, alert_no_shift, obs_no_shift = sequential_test(
            pred_seqs_no_shift,
            true_seqs_no_shift,
            delta,
            tol,
            ub_source,
            method=lb_method,
        )
        num_detections["no_shift"] = (
            num_detections["no_shift"] + 1
            if np.any(alert_no_shift)
            else num_detections["no_shift"]
        )

        # covariate shift
        lb_cov_shift, alert_cov_shift, obs_cov_shift = sequential_test(
            pred_seqs_cov_shift,
            true_seqs_cov_shift,
            delta,
            tol,
            ub_source,
            method=lb_method,
        )
        num_detections["cov_shift"] = (
            num_detections["cov_shift"] + 1
            if np.any(alert_cov_shift)
            else num_detections["cov_shift"]
        )

        # gradual concept shift
        lb_gcon_shift, alert_gcon_shift, obs_gcon_shift = sequential_test(
            pred_seqs_gcon_shift,
            true_seqs_gcon_shift,
            delta,
            tol,
            ub_source,
            method=lb_method,
        )
        num_detections["gcon_shift"] = (
            num_detections["gcon_shift"] + 1
            if np.any(alert_gcon_shift)
            else num_detections["gcon_shift"]
        )

        # abrupt concept shift
        lb_acon_shift, alert_acon_shift, obs_acon_shift = sequential_test(
            pred_seqs_acon_shift,
            true_seqs_acon_shift,
            delta,
            tol,
            ub_source,
            method=lb_method,
        )
        num_detections["acon_shift"] = (
            num_detections["acon_shift"] + 1
            if np.any(alert_acon_shift)
            else num_detections["acon_shift"]
        )

        #
        plt.figure(0)
        plt.plot(ns, lb_no_shift, color="blue", alpha=0.3)
        plt.plot(
            ns, obs_no_shift, color="blue", alpha=0.3, linestyle="--", linewidth=0.3
        )

        plt.figure(1)
        plt.plot(ns, lb_cov_shift, color="green", alpha=0.3)
        plt.plot(
            ns, obs_cov_shift, color="green", alpha=0.3, linestyle="--", linewidth=0.3
        )

        plt.figure(2)
        plt.plot(ns, lb_gcon_shift, color="orange", alpha=0.3)
        plt.plot(
            ns, obs_gcon_shift, color="orange", alpha=0.3, linestyle="--", linewidth=0.3
        )

        plt.figure(3)
        plt.plot(ns, lb_acon_shift, color="red", alpha=0.3)
        plt.plot(
            ns, obs_acon_shift, color="red", alpha=0.3, linestyle="--", linewidth=0.3
        )

    # report statistics
    print("\nSample Size: ", sample_size)
    print("Expected number of false alerts: ", delta * num_tests)
    print("# detections/#tests")
    print(
        "no shift: ",
        num_detections["no_shift"],
        "/",
        num_tests,
        " = ",
        num_detections["no_shift"] / num_tests,
    )
    print(
        "cov shift: ",
        num_detections["cov_shift"],
        "/",
        num_tests,
        " = ",
        num_detections["cov_shift"] / num_tests,
    )
    print(
        "gcon shift: ",
        num_detections["gcon_shift"],
        "/",
        num_tests,
        " = ",
        num_detections["gcon_shift"] / num_tests,
    )
    print(
        "acon shift: ",
        num_detections["acon_shift"],
        "/",
        num_tests,
        " = ",
        num_detections["acon_shift"] / num_tests,
    )

    # show and save plots
    plt.figure(0)
    plt.savefig(os.path.join("figures", "no_shift_many" + lb_method + ".png"))

    plt.figure(1)
    plt.savefig(os.path.join("figures", "cov_shift_many" + lb_method + ".png"))

    plt.figure(2)
    plt.savefig(os.path.join("figures", "gcon_shift_many" + lb_method + ".png"))

    plt.figure(3)
    plt.savefig(os.path.join("figures", "acon_shift_many" + lb_method + ".png"))
    plt.show()
