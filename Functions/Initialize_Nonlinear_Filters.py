#  ============================================================================
#  Name        : Initialize_Nonlinear_Fitlers.py
#  Description : Helper functions needed to intialize the simulation environment
#                for all nonlinear filers (EKF, UKF, CKF, CDKF, EnKF, BPF, and MPF).
#  Author      : Alex Nguyen
#  Date        : August 2022
#  ============================================================================

# Import Packages
import numpy as np
from scipy import linalg

# Functions
def initializeSoopVector(n_soops, east_soop_seed=5, north_soop_seed=10):
    """
    This function initializes the SoOP's state vector which is composed of the position and clock error states.
    """

    # Random SoOP Locations
    np.random.seed(east_soop_seed)
    east_soop_location = (1500 - (0))*np.random.rand(n_soops, 1) + (0)

    np.random.seed(north_soop_seed)
    north_soop_location = (500 - (-500))*np.random.rand(n_soops, 1) + (-500)

    # Clock Error States
    clock_bias  = np.ones((n_soops, 1))
    clock_drift = 0.1 * np.ones((n_soops, 1))

    # Construct SoOP State Vector
    soop_state_vector = np.transpose(np.hstack((east_soop_location, north_soop_location, clock_bias, clock_drift)))
    return soop_state_vector

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
    This function will construct the matrices needed to implement a nonlinear filter.
    """
    
    # Initialize Parameters
    speed_of_light         = 299792458
    number_of_measurements = unknown_soops + partially_known_soops
    number_of_states       = 4 + 2*partially_known_soops + 4*unknown_soops
    count_partially_known  = 0
    count_unknown          = 0

    # Linear Transformation Matrix (i.e., Similarity Transformation)
    linear_transformation_matrix = np.zeros((number_of_states, number_of_states + 2))

    Tr  = np.hstack((np.eye(4), np.zeros((4, 2))))                                     # Transformation Matrix for Receiver's Position/Velocity States
    Trn = np.tile(np.zeros((4, 2)), (1, partially_known_soops))
    Trm = np.tile(np.zeros((4, 4)), (1, unknown_soops))
    linear_transformation_matrix[0:4, :] = np.hstack((Tr, Trn, Trm))

    for i_nz in range(number_of_measurements):                                         # Transformation Matrix for SoOP's Position/Clock Error States
        if i_nz < partially_known_soops:
            # Create Transformation Matrix
            Tsi = -np.eye(2)
            Tni = np.hstack((np.tile(np.zeros((2, 2)), (1, i_nz)), Tsi, np.tile(np.zeros((2, 2)), (1, partially_known_soops - (i_nz + 1))), 
                             np.tile(np.zeros((2, 4)), (1, unknown_soops))))
            Trclk = np.hstack((np.zeros((2, 4)), np.eye(2))) 
            linear_transformation_matrix[4 + 2*count_partially_known:4 + 2*count_partially_known + 2, :] = np.hstack((Trclk, Tni))
            
            # Update Counter
            count_partially_known += 1    

        elif i_nz >= partially_known_soops and i_nz <= number_of_measurements:
            # Create Transformation Matrix
            Tsi = linalg.block_diag(np.eye(2), -np.eye(2)) 
            Tmi = np.hstack((np.tile(np.zeros((4, 2)), (1, partially_known_soops)), 
                             np.tile(np.zeros((4, 4)), (i_nz - partially_known_soops)), Tsi, np.tile(np.zeros((4, 4)), (1, number_of_measurements - (i_nz + 1)))))
            Trclk = linalg.block_diag(np.zeros((2, 4)), np.eye(2))
            linear_transformation_matrix[4 + 2*partially_known_soops + 4*count_unknown:4 + 2*partially_known_soops + 4*(count_unknown + 1), :] = np.hstack((Trclk, Tmi))

            # Update Counter
            count_unknown += 1
            
        else:
            print('error')

    # Initialize Matrices
    Fi = np.empty((0, 0))
    Pi = np.empty((0, 0))
    Qi = np.empty((0, 0))

    for i_nz in range(number_of_measurements):                                         
        if i_nz < partially_known_soops:
            # State Transition 
            Fi = linalg.block_diag(Fi, clock_state_transition)

            # Process Covariance 
            Qi = linalg.block_diag(Qi, speed_of_light**2*soop_clock_process_noise)

            # Estimation Error Covariance
            Pi = linalg.block_diag(Pi, partially_known_soop_estimation_error_covariance)

        elif i_nz >= partially_known_soops and i_nz <= number_of_measurements:
            # State Transition 
            Fi = linalg.block_diag(Fi, soop_state_transition)

            # Process Covariance 
            Qi = linalg.block_diag(Qi, linalg.block_diag(np.finfo(float).eps*np.eye(2), speed_of_light**2*soop_clock_process_noise))

            # Estimation Error Covariance
            Pi = linalg.block_diag(Pi, unknown_soop_estimation_error_covariance)

    # State Transition Matrix
    state_transition_matrix = linalg.block_diag(dynamics_state_transition, Fi)
    
    # Input Matrix
    input_matrix = np.zeros(((len(state_transition_matrix)), 2))

    # Estimation Error Covariance Matrix
    estimation_error_covariance_matrix = linear_transformation_matrix @ linalg.block_diag(receiver_estimation_error_covariance, Pi) @ linear_transformation_matrix.transpose()

    # Process Covariance Matrix
    process_covariance_matrix = linear_transformation_matrix @ linalg.block_diag(dynamics_process_noise, speed_of_light**2*receiver_clock_process_noise, Qi) @ linear_transformation_matrix.transpose()
    
    # Measurement Covariance Matrix
    measurement_covariance_matrix = measurement_noise*np.eye(number_of_measurements)

    return linear_transformation_matrix, state_transition_matrix, input_matrix, estimation_error_covariance_matrix, process_covariance_matrix, measurement_covariance_matrix

def constructStateVector(partially_known_soops, unknown_soops, 
                         receiver_state_vector, soop_state_vector, 
                         estimation_error_covariance_matrix, linear_transformation_matrix, 
                         estimate_state_seed=7):
    """
    This function will construct the initial true and estimated state vectors for the nonlinear filter.
    """
    
    # Initialize
    number_of_measurements = partially_known_soops + unknown_soops
    clock_error_terms      = np.vstack((soop_state_vector[2, :], soop_state_vector[3, :]))
    
    # Preallocate
    ith_soop_state_vector = []

    # Construct SoOP State Vector
    for i in range(number_of_measurements):
        if i < partially_known_soops:
            # i-th SoOP Clock Error State Vector
            ith_soop_state_vector = np.append(ith_soop_state_vector, clock_error_terms[:, i])
            
        elif i >= partially_known_soops and i <= number_of_measurements:
            # i-th SoOP State Vector
            ith_soop_state_vector = np.append(ith_soop_state_vector, soop_state_vector[:, i])
        
    # True State Vector
    true_state_vector_untransformed = np.append(receiver_state_vector, ith_soop_state_vector)
        
    # Linear Transformation on True State Vector
    true_state_vector = linear_transformation_matrix @ true_state_vector_untransformed

    # Estimated State Vector
    np.random.seed(estimate_state_seed)
    estimate_state_vector = np.random.multivariate_normal(true_state_vector, estimation_error_covariance_matrix).transpose()
    
    # Reshape Dimensions
    true_state_vector     = true_state_vector.reshape(len(true_state_vector), 1)
    estimate_state_vector = estimate_state_vector.reshape(len(estimate_state_vector), 1)

    # Input State Vector
    input_state_vector = np.zeros((2, 1))
    
    return true_state_vector, estimate_state_vector, input_state_vector
    
    
def truePseudorangeMeasurements(true_state_vector, soop_state_vector, partially_known_soops, unknown_soops):
    """
    This function will compute the pseudorange measurements at a specific time instance for the nonlinear filter.
    """
    
    # Initialize
    number_of_measurements = partially_known_soops + unknown_soops
    partially_known_index  = (4 + 1) - 1 
    unknown_index          = (4 + 2*partially_known_soops + 1) - 1
    
    # Preallocate
    pseudorange_measurement = np.zeros((number_of_measurements, 1))
    
    for i in range(number_of_measurements):
        if i < partially_known_soops:
            # Compute Pseudorange Measurement
            distance                   = linalg.norm(true_state_vector[0:2] - soop_state_vector[0:2, i].reshape(2, 1))
            pseudorange_measurement[i] = distance + true_state_vector[partially_known_index]
            
            # Update Index
            partially_known_index += 2
            
        elif i >= partially_known_soops and i <= number_of_measurements:
            # Compute Pseudorange Measurement
            distance                   = linalg.norm(true_state_vector[0:2] - true_state_vector[unknown_index:unknown_index + 2])
            pseudorange_measurement[i] = distance + true_state_vector[unknown_index + 2]
            
            # Update Index
            unknown_index += 4
    
    return pseudorange_measurement    