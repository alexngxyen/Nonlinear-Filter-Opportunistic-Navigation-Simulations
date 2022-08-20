# Import Packages
from socket import TIPC_CONN_TIMEOUT
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

def matrixInitialization(dynamics_state_transition, soop_state_transition, clock_state_transition,
                         dynamics_process_noise, receiver_clock_process_noise, soop_clock_process_noise, 
                         receiver_estimation_error_covariance, unknown_soop_estimation_error_covariance, partially_known_soop_estimation_error_covariance, 
                         measurement_noise, partially_known_soops, unknown_soops):
    """
    This function will construct the matrices needed to implement the nonlinear filter.
    """
    
    # Initialize Parameters
    speed_of_light         = 299792458
    number_of_measurements = unknown_soops + partially_known_soops
    number_of_states       = 4 + 2*partially_known_soops + 4*unknown_soops

    # Linear Transformation Matrix (i.e., Similarity Transformation)
    linear_transformation_matrix = np.zeros((number_of_states, number_of_states + 2))

    Tr = np.hstack((np.eye(4), np.zeros((4, 2)) ))                                     # Transformation Matrix for Receiver's Position/Velocity States
    Trn = np.tile(np.zeros((4, 2)), (1, partially_known_soops))
    Trm = np.tile(np.zeros((4, 4)), (1, unknown_soops))
    linear_transformation_matrix[0:4, :] = np.hstack((Tr, Trn, Trm))

    for i_nz in range(number_of_measurements):                                         # Transformation Matrix for SoOP's Position/Clock Error States
        if i_nz <= partially_known_soops:
            print(i_nz)
            Tsi = -np.eye(2)
            Tni = np.hstack((np.tile(np.zeros((2, 2)), (1, i_nz)), Tsi, np.tile(np.zeros((2, 2)), (1, partially_known_soops - (i_nz + 1))), 
                            np.tile(np.zeros((2, 4)), (1, unknown_soops))))
            Trclk = np.hstack((np.zeros((2, 4)), np.eye(2))) 
            linear_transformation_matrix[4 + 2*i_nz:4 + 2*i_nz + 2, :] = np.hstack((Trclk, Tni))    

        elif i_nz > partially_known_soops & i_nz <= number_of_measurements:
            Tsi = linalg.block_diag(np.eye(2), -np.eye(2)) 
            Tmi = np.hstack((np.tile(np.zeros((4, 2)), (1, partially_known_soops)), 
                            np.tile(np.zeros((4, 4)), (1, (i_nz + 1) - (partially_known_soops + 1))), Tsi, np.tile(np.zeros((4, 4)), (1, number_of_measurements - (i_nz + 1)))))
            Trclk = linalg.block_diag(np.zeros((2, 4)), np.eye(2))
            linear_transformation_matrix[4 + 2*partially_known_soops + 4*(i_nz + 1):4 + 2*partially_known_soops + 4*(i_nz + 1) + 4, :] = np.hstack((Trclk, Tmi))

        else:
            print('error')

    # Initialize Matrices
    Fi = np.empty(2, 2)
    Pi = Fi
    Qi = Fi

    for i_nz in range(number_of_measurements):                                         
        if i_nz <= partially_known_soops:
            # State Transition 
            Fi = linalg.block_diag(Fi, clock_state_transition)

            # Process Covariance 
            Qi = linalg.block_diag(Qi, c**2*soop_clock_process_noise)

            # Estimation Error Covariance
            Pi = linalg.block_diag(Pi, partially_known_soop_estimation_error_covariance)

        elif i_nz > partially_known_soops & i_nz <= number_of_measurements:
            # State Transition 
            Fi = linalg.block_diag(Fi, soop_state_transition)

            # Process Covariance 
            Qi = linalg.block_diag(Qi, linalg.block_diag(np.finfo(float).eps*np.eye(2), c**2*soop_clock_process_noise))

            # Estimation Error Covariance
            Pi = linalg.block_diag(Pi,unknown_soop_estimation_error_covariance)

    # State Transition Matrix
    state_transition_matrix = linalg.block_diag(dynamics_state_transition, Fi)
    input_matrix = np.zeros((len(state_transition_matrix)), 2)

    # Estimation Error Covariance Matrix
    estimation_error_covariance_matrix = linear_transformation_matrix @ linalg.block_diag(receiver_estimation_error_covariance, Pi) @ linear_transformation_matrix.transpose()

    # Process Covariance Matrix
    process_covariance_matrix = linear_transformation_matrix @ linalg.block_diag(dynamics_process_noise, c**2*receiver_clock_process_noise, Qi) @ linear_transformation_matrix.transpose()
    
    # Measurement Covariance Matrix
    measurement_covariance_matrix = measurement_noise*np.eye(number_of_measurements)

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
# Dynamics 
qx = 0.25                                                # East Position Process Noise
qy = qx                                                  # North Position Process Noise

# Receiver Clock 
S_wtr    = h0_rx/2
S_wtrdot = 2*math.pi**2*hneg2_rx

# SoOP Clock
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
Qpv    = np.array([[qx*T**3/3, 0, qx*T**2/2, 0], [0, qy*T**3/3, 0, qy*T**2/2], [qx*T**2/2, 0, qx*T, 0], [0, qy*T**2/2, 0, qy*T]])
Qclk_r = np.array([[S_wtr*T + S_wtrdot*T**3/3, S_wtrdot*T**2/2], [S_wtrdot*T**2/2, S_wtrdot*T]])

""" Initialize Nonlinear Filter"""
# Nonlinear Filter Parameters
measurement_noise = 25                            
P_rx0  = linalg.block_diag(5**2*np.eye(2), 1**2*np.eye(2), 30**2, 0.3**2)                # Receiver's Initial Estimation Error Covariance Matrix
P_s0   = linalg.block_diag(1e3*np.eye(2), 30**2, 0.3**2)                                 # Unknown SoOP's Initial Estimation Error Covariance Matrix
P_clk0 = linalg.block_diag(30**2, 0.3**2)                                                # Partially-Known SoOP's Initial Estimation Error Covariance Matrix

# Construct Necessary Matrices (e.g., T, F, G, P, Q, R)
T, F, G, P, Q, R = matrixInitialization(Fpv, Fs, Fclk, Qpv, Qclk_r, Qclk_s, P_rx0, P_s0, P_clk0, measurement_noise, n, m)

# Process and Measurement Noise
q = linalg.cholesky(Q, lower=True)
r = linalg.cholesky(R, lower=True) 

# Extended Kalman Filter States
