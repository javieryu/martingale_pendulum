# martingale_pendulum

# Model Learning
Trajectory data from which to train both the single and double pendulum models in generated in ```simulator.py```.
The learned models are then trained in ```train_model.py```.
Functions for loading and evaluating a trained model are defined in ```test_model.py```. 
Also in ```test_model.py``` are functions to visualize the distribution of training data as well as the learned phase diagram of the dynamics.

# Experiments
Sample trajectories for each experiment are created using ```gen_experiment_data.py```.
The experiments that take full trajectories as samples are run in ```detector_traj.py```.
The experiments that take single step predictions as samples are run in ```detector.py```.

# References
The methods used in this repository were developed in the paper, <em>Tracking the risk of a deployed model and detecting harmful distribution shifts</em> by Aleksandr Podkopaev and Aaditya Ramdas.
