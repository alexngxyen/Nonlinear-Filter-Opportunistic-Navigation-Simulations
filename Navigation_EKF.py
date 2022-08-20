# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import linalg

""" Functions """
def initializeSoopVector(n_soops, east_soop_seed=1, north_soop_seed=2):
    """
    This function initializes the SoOP's state vector which is composed of the position and clock error states.
    """

    # Random SoOP Locations
    np.random.seed(east_soop_seed)
    east_soop_location = (1000 - (-250))*np.random.rand(n_soops, 1) + (-250)

    np.random.seed(north_soop_seed)
    north_soop_location = (500 - (-500))*np.random.rand(n_soops, 1) + (-500)

    # Clock Error States
    clock_bias  = np.ones((n_soops, 1))
    clock_drift = 0.1 * np.ones((n_soops, 1))

    # Construct SoOP State Vector
    xs0 = np.transpose(np.hstack((east_soop_location, north_soop_location, clock_bias, clock_drift)))
    return xs0

def clockQuality(clock_type):
    """
    This function obtains the power-law coefficients for the clock bias and drift process noise power spectra of a specific clock quality type.
    """

    if clock_type == ('CSAC'):
        white_frequency_coefficient       = 2.0e-20
        frequency_random_walk_coefficient = 4.0e-29   
    elif clock_type == ('Best OCXO'):
        white_frequency_coefficient       = 2.6e-22
        frequency_random_walk_coefficient = 4.0e-26  
    elif clock_type == ('Typical OCXO'):
        white_frequency_coefficient       = 8.0e-20
        frequency_random_walk_coefficient = 4.0e-23 
    elif clock_type == ('Typical TCXO'):
        white_frequency_coefficient       = 9.4e-20
        frequency_random_walk_coefficient = 3.8e-21 
    elif clock_type == ('Worst TCXO'):
        white_frequency_coefficient       = 2.0e-19
        frequency_random_walk_coefficient = 2.0e-20 
    else:
        print('clock quality type not recognized.')
    return white_frequency_coefficient, frequency_random_walk_coefficient

# define variables nicely
def matrixInitialization(dynamics_state_transition, soop_state_transition, clock_state_transition,
                         dynamics_process_noise, receiver_clock_process_noise, soop_clock_process_noise, 
                         receiver_estimation_error_covariance, unknown_soop_estimation_error_covariance, partially_known_soop_estimation_error_covariance, 
                         measurement_noise, partially_known_soops, unknown_soops):
    """
    This function will construct the matrices needed to implement the nonlinear filter.
    """
    
    # Linear Transformation Matrix (i.e., Similarity Transformation)


    # Initialize Matrices


    # State Transition Matrix


    # Process Noise Matrix


    # Estimation Error Covariance Matrix


    return linear_transformation_matrix, state_transition_matrix, input_matrix, estimation_error_covariance_matrix, process_covariance_matrix, measurement_covariance_matrix

""" Simulation Parameters """
# SoOPs
m  = 0                                                   # Unknown SoOPs
n  = 10                                                  # Partially-Known (Position States) SoOPs 
nz = n + m                                               # Total Number of SoOP Measurements
nx = 4 + 2*n + 4*m                                       # Total Number of States

# Time
c = 299792458                                            # Speed of Light [m/s]
T = 10e-2                                                # Sampling Period [s]
t = np.arange(0, 50 + T, T)                              # Experiment Time Duration [s]
simulation_length = np.size(t)                           # Simulation Time Length [samples]

# Initialize Receiver Parameters
x_rx0 = np.array([0, 50, 15, -5, 100, 1]).reshape(6, 1)  # State Vector      
h0_rx, hneg2_rx = clockQuality('Typical OCXO')           # Typical Oven-Controlled Crystal Oscillator (OCXO)

# Initialize SoOP Parameters
x_s0 = initializeSoopVector(nz)                          # State Vector                              
h0_s, hneg2_s = clockQuality('Typical OCXO')             # Typical Oven-Controlled Crystal Oscillator (OCXO) 

""" Power Spectral Density """
# Dynamics Process Noise
qx = 0.25                                                # East Position Process Noise
qy = qx                                                  # North Position Process Noise

# Receiver Process Noise
S_wtr    = h0_rx/2
S_wtrdot = 2*math.pi**2*hneg2_rx

# SoOP Process Noise
S_wts    = h0_s/2; 
S_wtsdot = 2*math.pi**2*hneg2_s

""" SoOP Dynamics """
# Clock Error State Transition Matrix
Fclk = np.array([[1, T], [0, 1]])

# SoOP State Transition Matrix
Fs = linalg.block_diag(np.eye(2), Fclk)

# SoOP Process Covariance
Qclk_s = np.array([[S_wts*T + S_wtsdot*T**3/3, S_wtsdot*T**2/2], [S_wtsdot*T**2/2, S_wtsdot*T]])

""" Receiver Dynamics """
# Velocity Random Walk State Transition Matrix
Fpv = np.bmat([[np.eye(2), np.eye(2)*T], [np.zeros((2, 2)), np.eye(2)]])

# Receiver State Transition Matrix
Fr = linalg.block_diag(Fpv, Fclk)

# Receiver Process Covariance
Qpv   = np.array([[qx*T**3/3, 0, qx*T**2/2, 0], [0, qy*T**3/3, 0, qy*T**2/2], [qx*T**2/2, 0, qx*T, 0], [0, qy*T**2/2, 0, qy*T]])
Qpv_r = np.array([[S_wtr*T + S_wtrdot*T**3/3, S_wtrdot*T**2/2], [S_wtrdot*T**2/2, S_wtrdot*T]])

""" Initialize Nonlinear Filter"""
# Nonlinear Filter Parameters
measurement_noise = 25                            
P_rx0  = linalg.block_diag(5**2*np.eye(2), 1**2*np.eye(2), 30**2, 0.3**2)                # Receiver's Initial Estimation Error Covariance Matrix
P_s0   = linalg.block_diag(1e3*np.eye(2), 30**2, 0.3**2)                                 # Unknown SoOP's Initial Estimation Error Covariance Matrix
P_clk0 = linalg.block_diag(30**2, 0.3**2)                                                # Partially-Known SoOP's Initial Estimation Error Covariance Matrix

# Construct Necessary Matrices (e.g., T, F, G, P, Q, R)
