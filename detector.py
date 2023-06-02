import numpy as np
import confseq

"""
This file has code for
	1. upper bounding the source risk
	2. lower bounding the target risk

The PM-H, PM-EB, and Betting confidence sequences assume that target data consists of iid samples from an unknown distribution
The Betting bound is currently not implemented
The CM-EB confidence sequence assumes that the target data is a "predictable sequence", this is what we use for the time-varying distribution shift
"""


def source_risk_ub(pred_seq, true_seq, delta):
	"""
	Use PM-EB method to upper bound source risk
	Risk is the expected value of absolute error
	pred_seq: array of network predictions where pred_seq.shape = (# predictions, prediction dimension)
	true_seq: array of true labels corresponding to each network prediction
	delta: upper bound holds with probability 1-delta
	"""
	risks = np.abs(pred_seq - true_seq) # consider absolute error as risk metric
	_, ubs = bounds_pm_eb(risks, delta)

	ub_source = ubs[-1]

	return ub_source



def sequential_test(pred_seq, true_seq, delta, tol, ub_source, method="PM-H"):
	"""
	pred_seq: array of network predictions where pred_seq.shape = (# predictions, prediction dimension)
	true_seq: array of true labels corresponding to each network prediction
	delta: target risk lower bounds holds with probability 1-delta
	tol: how much slack between source and target risk to allow before raising alert
	ub_source: upper bound on the source risk
	Risk is expected value of absolute error
	"""

	risks = np.abs(pred_seq - true_seq) # consider absolute error as risk metric

	if method == "PM-H":
		lbs_target, _ = bounds_pm_h(risks, delta)
	elif method == "PM-EB":
		lbs_target, _ = bounds_pm_eb(risks, delta)
	elif method == "Betting":
		lbs_target = bounds_betting(risks, delta)
	elif method == "CM-EB":
		lbs_target = bounds_cm_eb(risks, delta)

	alerts = lbs_target > (ub_source + tol) # make boolean array of if alert is raised

	return lbs_target, alerts




def bounds_pm_h(Zs, delta):
	"""
	Predictably-mixed Hoeffding's confidence sequence
	"""
	T = len(Zs)
	lb = np.zeros(T)
	ub = np.zeros(T)
	lambdas = np.zeros(T)
	psis = np.zeros(T)
	for idx,t in enumerate(range(1,T+1)):
		lambdas[idx] = np.min(1, np.sqrt((8*np.log(1/delta)) / (t*np.log(t+1))))
		psis[idx] = lambdas[t]**2 / 8

		# compute first term in lb/ub
		term1 = np.dot(lambdas, Zs) / np.sum(lambdas)

		# compute second term in lb/ub
		term2 = (np.log(1/delta) + np.sum(psis)) / np.sum(lambdas)

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
	for idx,t in enumerate(range(1,T+1)):
		mus[idx] = (0.5 + np.sum(Zs[:idx+1])) / (t+1)
		var = (0.25 + np.sum(((Zs - mus)**2)[:idx+1])) / (t+1)
		lambdas[idx] = np.min(c, np.sqrt((2*np.log(1/delta)) / (var * t * np.log(t+1))))
		
		if t == 1:
			vs[idx] = 4 * (Zs[idx])**2 # assume initial empirical mean is zero
		else:
			vs[idx] = 4 * (Zs[idx] - mus[idx-1])**2
		psis[idx] = (-np.log(1 - lambdas[idx]) - lambdas[idx]) / 4
		
		# compute first term in lb/ub
		term1 = np.dot(lambdas, Zs) / np.sum(lambdas)

		# compute second term in lb/ub
		term2 = (np.log(1/delta) + np.dot(vs, psis)) / np.sum(lambdas)

		
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
	v_opt = len(Zs) / 2 # make bound tightest halfway through traj
	lbs = confseq.conjmix_empbern_lower_cs(Zs, v_opt, alpha=delta, running_intersection=False)
	return lbs



