#  ============================================================================
#  Name        : Unscented_Kalman_Filter.py
#  Description : Helper functions needed to run the 'UKF_Simulation.py' code  
#  Author      : Alex Nguyen
#  Date        : August 2022
#  ============================================================================
#  Notes       : - If you have any questions or comments about this code, please 
#                  message the following email address "alexaan2@uci.edu".
#  ============================================================================

# Import Packages
import numpy as np
from scipy import linalg

""" EDIT THE FUNCTIONS BELOW HERE!!"""

# Functions
def predictionStep(corrected_state_vector, input_vector, state_transition_matrix, input_matrix, process_covariance_matrix, estimation_error_covariance_matrix):
    """
    This function will perform the time-update (prediction step) of the extended Kalman filter.
    """
    
    # State Vector Prediction
    predicted_state_vector = state_transition_matrix @ corrected_state_vector + input_matrix @ input_vector 
    
    # Estimation Error Covariance Prediction
    predicted_estimation_error_covariance = state_transition_matrix @ estimation_error_covariance_matrix @ state_transition_matrix.transpose() + process_covariance_matrix
    predicted_estimation_error_covariance = 0.5 * (predicted_estimation_error_covariance + predicted_estimation_error_covariance.transpose())                                  # Ensure symmetry

    return predicted_state_vector, predicted_estimation_error_covariance
    
def estimatedPseudorangeMeasurements(estimate_state_vector, soop_state_vector, partially_known_soops, unknown_soops):
    """
    This function will compute the pseudorange measurements at a specific time instance for the nonlinear filter.
    """
    
    # Initialize
    number_of_measurements = partially_known_soops + unknown_soops
    partially_known_index  = (4 + 1) - 1 
    unknown_index          = (4 + 2*partially_known_soops + 1) - 1
    clock_jacobian         = np.array([1, 0]).reshape(1, 2)
    
    # Preallocate
    jacobian_matrix                  = np.zeros((number_of_measurements, len(estimate_state_vector)))
    estimate_pseudorange_measurement = np.zeros((number_of_measurements, 1))
    
    for i in range(number_of_measurements):
        if i < partially_known_soops:
            # Compute Estimate Pseudorange Measurement
            distance                            = linalg.norm(estimate_state_vector[0:2] - soop_state_vector[0:2, i].reshape(2, 1))
            estimate_pseudorange_measurement[i] = distance + estimate_state_vector[partially_known_index]
            
            # Construct Jacobian Matrix
            receiver_jacobian_component = np.hstack((np.hstack(((estimate_state_vector[0] - soop_state_vector[0, i]) / distance, (estimate_state_vector[1] - soop_state_vector[1, i]) / distance)).reshape(1, 2),
                                                     np.zeros((1, 2))))
            soop_jacobian_component     = np.hstack((np.tile(np.zeros((1, 2)), (1, i)), clock_jacobian, np.tile(np.zeros((1, 2)), (1, partially_known_soops - (i + 1))), 
                                                     np.tile(np.zeros((1, 4)), (1, unknown_soops))))
            jacobian_matrix[i:i+1, :]   = np.hstack((receiver_jacobian_component, soop_jacobian_component))
            
            # Update Index
            partially_known_index += 2
            
        elif i >= partially_known_soops and i <= number_of_measurements:
            # Compute Estimate Pseudorange Measurement
            distance                            = np.linalg.norm(estimate_state_vector[0:2] - estimate_state_vector[unknown_index:unknown_index + 2])
            estimate_pseudorange_measurement[i] = distance + estimate_state_vector[unknown_index + 2]
            
            # Construct Jacobian Matrix
            receiver_jacobian_component = np.hstack((np.hstack(((estimate_state_vector[0] - estimate_state_vector[unknown_index]) / distance, (estimate_state_vector[1] - estimate_state_vector[unknown_index + 1]) / distance)).reshape(1, 2), 
                                                     np.zeros((1, 2))))
            ith_soop_jacobian_component = np.hstack((np.hstack((-(estimate_state_vector[0] - estimate_state_vector[unknown_index]) / distance, -(estimate_state_vector[1] - estimate_state_vector[unknown_index + 1]) / distance)).reshape(1, 2),
                                                     clock_jacobian))
            soop_jacobian_component     = np.hstack((np.tile(np.zeros((1, 2)), (1, partially_known_soops)), 
                                                     np.tile(np.zeros((1, 4)), (1, i - partially_known_soops)), ith_soop_jacobian_component, np.tile(np.zeros((1, 4)), (1, number_of_measurements - (i + 1)))))
            jacobian_matrix[i:i+1, :]   = np.hstack((receiver_jacobian_component, soop_jacobian_component))
                        
            # Update Index
            unknown_index += 4
    
    return estimate_pseudorange_measurement, jacobian_matrix        

def correctionStep(predicted_state_vector, predicted_estimation_error_covariance_matrix, 
                   true_pseudorange_measurement, estimate_pseudorange_measurement, jacobian_matrix, measurement_covariance_matrix):
    """
    This function will perform the measurement-update (correction step) of the extended Kalman filter.
    """
    
    # Measurement Residual
    innovation = true_pseudorange_measurement - estimate_pseudorange_measurement
    
    # Innovation Covariance Matrix
    innovation_covariance_matrix = jacobian_matrix @ predicted_estimation_error_covariance_matrix @ jacobian_matrix.transpose() + measurement_covariance_matrix
    
    # Kalman Gain Matrix
    kalman_gain_matrix = predicted_estimation_error_covariance_matrix @ jacobian_matrix.transpose() @ linalg.inv(innovation_covariance_matrix)
    
    # State Vector Correction
    corrected_state_vector = predicted_state_vector + kalman_gain_matrix @ innovation
    
    # Estimation Error Covariance Correction
    jordan_form_matrix                    = np.eye(len(corrected_state_vector)) - kalman_gain_matrix @ jacobian_matrix
    corrected_estimation_error_covariance = jordan_form_matrix @ predicted_estimation_error_covariance_matrix @ jordan_form_matrix.transpose() + kalman_gain_matrix @ measurement_covariance_matrix @ kalman_gain_matrix.transpose()
    corrected_estimation_error_covariance = 0.5 * (corrected_estimation_error_covariance + corrected_estimation_error_covariance.transpose())                # Ensure symmetry

    return corrected_state_vector, corrected_estimation_error_covariance