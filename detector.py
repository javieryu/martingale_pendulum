import numpy as np

"""
Detector related code goes in here.
"""

def sequential_test(pred_seq, true_seq, tol, ub_source):
	T = len(true_seq)
	lb_target = np.zeros(T)
	alert = np.zeros(T)

	for t in range(T):
		lb_target[t] = target_risk_lb(pred_seq[0:t+1], true_seq[0:t+1], method="PM-H")
		alert = lb_target > ub_source + tol

	return lb_target, alert
		


def target_risk_lb(pred_seq, true_seq, method="PM-H"):
	"""
	computes a lower confidence sequence for the target risk
	method: "PM-H", "PM-EB", "Betting", "CM-EB" 
	"""
	risks = pred_seq - true_seq

	if method == "PM-H":
		lb = lb_pm_h(risks)
	elif method == "PM-EB":
		lb = lb_pm_eb(risks)
	elif method == "Betting":
		lb = lb_betting(risks)
	elif method == "CM-EB":
		lb = lb_cm_eb(risks)

	return lb




def lb_pm_h(Z):
	"""
	Predictably-mixed Hoeffding's confidence sequence
	"""
	None
	return lb


def lb_pm_eb(Z):
	"""
	Predictably-mixed empirical-Bernstein confidence sequence
	"""
	None
	return lb


def lb_betting(Z):
	"""
	Betting-based confidence sequence
	"""
	None
	return lb


def lb_cm_eb(Z):
	"""
	Conjugate-mixture empirical-Bernstein confidence sequence
	"""
	None
	return lb


def source_risk_ub():
	"""
	Use standard results for bounding iid source risk
	"""
	None
	return ub