import numpy as np
import confseq

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




def lb_pm_h(Zs, delta):
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


def lb_pm_eb(Zs, delta):
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
		var = (0.25 + np.sum((Zs - mus)**2[:idx+1])) / (t+1)
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




def lb_betting(Z):
	"""
	Betting-based confidence sequence.
	Might not implement.
	"""
	None
	return None





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