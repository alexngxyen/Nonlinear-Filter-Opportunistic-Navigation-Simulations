#  ============================================================================
#  Name        : Central_Difference_Kalman_Filter.py
#  Description : Helper functions needed to run the 'CDKF_Simulation.py' code  
#  Author      : Alex Nguyen
#  Date        : August 2022
#  ============================================================================

# Import Packages
import numpy as np
from scipy import linalg

# Functions
def predictionStep(number_of_sigma_points, number_of_states, sigma_point_parameter, mean_weights, covariance_weight_one, covariance_weight_two, 
                   corrected_state_vector, corrected_estimation_error_covariance_matrix, state_transition_matrix, process_covariance_matrix):
    """
    This function will perform the time-update (prediction step) for the central difference Kalman filter.
    """
    
    # Preallocate
    sigma_points = np.zeros((number_of_states, number_of_sigma_points))
    
    # Matrix Square Root via Cholesky Decomposition
    square_root_covariance = linalg.cholesky(corrected_estimation_error_covariance_matrix, lower=True) 
    
    # Compute 2*nx + 1 Sigma Points
    sigma_points[:, 0:1] = corrected_state_vector                                                                      # Sigma Point 1
    
    for i in range(number_of_states):
        # Sigma Points 2, ..., nx
        sigma_points[:, i+1:i+2] = corrected_state_vector + sigma_point_parameter * square_root_covariance[:, i:i+1]
        
        # Sigma Points nx + 1, nx + 2, ..., 2*nx + 1
        sigma_points[:, i+1+number_of_states:i+2+number_of_states] = corrected_state_vector - sigma_point_parameter * square_root_covariance[:, i:i+1]
    
    # Propogate Sigma Points Through Dynamics Model
    sigma_points_propogated = state_transition_matrix @ sigma_points

    # Predicted Mean (State Estimate)
    predicted_state_vector = sigma_points_propogated @ mean_weights.transpose()
    
    # Predicted Covariance Matrix (Estimation Error Covariance)
    estimation_error_covariance_term_one  = sigma_points_propogated[:, 1:number_of_states+1] - sigma_points_propogated[:, number_of_states+1:number_of_sigma_points]
    estimation_error_covariance_term_two  = sigma_points_propogated[:, 1:number_of_states+1] + sigma_points_propogated[:, number_of_states+1:number_of_sigma_points] - 2*sigma_points_propogated[:, 0:1]
    predicted_estimation_error_covariance = covariance_weight_one * (estimation_error_covariance_term_one @ estimation_error_covariance_term_one.transpose()) + covariance_weight_two * (estimation_error_covariance_term_two @ estimation_error_covariance_term_two.transpose()) + process_covariance_matrix
    
    # Enforce Covariance Symmetry
    predicted_estimation_error_covariance = 0.5*(predicted_estimation_error_covariance + predicted_estimation_error_covariance.transpose())                                             
  
    return predicted_state_vector, predicted_estimation_error_covariance

def estimatedPseudorangeMeasurements(number_of_sigma_points, number_of_states, unknown_soops, partially_known_soops, sigma_point_parameter, mean_weights, 
                                     predicted_state_vector, predicted_estimation_error_covariance_matrix, soop_state_vector):
    """
    This function will compute the estimated pseudorange measurements and sigma points at a specific time instance for the central difference Kalman filter.
    """

    # Preallocate
    sigma_points_new         = np.zeros((number_of_states, number_of_sigma_points))
    sigma_points_measurement = np.zeros((partially_known_soops + unknown_soops, number_of_sigma_points))
    
    # Matrix Square Root via Cholesky Decomposition
    square_root_covariance = linalg.cholesky(predicted_estimation_error_covariance_matrix, lower=True) 

    # Compute 2*nx + 1 Sigma Points
    sigma_points_new[:, 0:1] = predicted_state_vector                                                                   # Sigma Point 1
    
    for i in range(number_of_states):
        # Sigma Points 2, ..., nx
        sigma_points_new[:, i+1:i+2] = predicted_state_vector + sigma_point_parameter * square_root_covariance[:, i:i+1]
        
        # Sigma Points nx + 1, nx + 2, ..., 2*nx + 1
        sigma_points_new[:, i+1+number_of_states:i+2+number_of_states] = predicted_state_vector - sigma_point_parameter * square_root_covariance[:, i:i+1]
    
    # Propogate Sigma Points Through Measurement Model
    number_of_measurements = partially_known_soops + unknown_soops
    partially_known_index  = (4 + 1) - 1 
    unknown_index          = (4 + 2*partially_known_soops + 1) - 1
    
    for i in range(number_of_measurements):
        # Partially-Known SoOPs
        if i < partially_known_soops:
            # Evaluate Sigma Points 
            distance   = linalg.norm((sigma_points_new[:2, :] - np.tile(soop_state_vector[:2, i].reshape(2, 1), (1, number_of_sigma_points))).reshape(2, number_of_sigma_points), axis=0).reshape(1, number_of_sigma_points)
            clock_bias = sigma_points_new[partially_known_index:partially_known_index+1, :]
            sigma_points_measurement[i:i+1, :] = distance + clock_bias
            
            # Update Index
            partially_known_index += 2
            
        elif i >= partially_known_soops and i <= number_of_measurements:
            # Evaluate Sigma Points
            distance   = linalg.norm((sigma_points_new[:2, :] - sigma_points_new[unknown_index:unknown_index+2, :]).reshape(2, number_of_sigma_points), axis=0).reshape(1, number_of_sigma_points)
            clock_bias = sigma_points_new[unknown_index+2:unknown_index+3, :]
            sigma_points_measurement[i:i+1, :] = distance + clock_bias
            
            # Update Index
            unknown_index += 4
            
    # Esimate Pseudorange Measurements
    estimate_pseudorange_measurement = sigma_points_measurement @ mean_weights.transpose()
    
    return sigma_points_measurement, estimate_pseudorange_measurement

def correctionStep(number_of_sigma_points, number_of_states, covariance_weight_one, covariance_weight_two, estimate_pseudorange_measurement, pseudorange_measurement, sigma_points_measurement, 
                   predicted_state_vector, predicted_estimation_error_covariance_matrix, measurement_covariance_matrix):
    """
    This function will perform the measurement-update (correction step) for the central difference Kalman filter.
    """
    
    # Measurement-Measurement Covariance Matrix
    measurement_measurement_covariance_matrix_one = sigma_points_measurement[:, 1:number_of_states+1] - sigma_points_measurement[:, number_of_states+1:number_of_sigma_points]
    measurement_measurement_covariance_matrix_two = sigma_points_measurement[:, 1:number_of_states+1] + sigma_points_measurement[:, number_of_states+1:number_of_sigma_points] - 2*sigma_points_measurement[:, 0:1]
    measurement_measurement_covariance_matrix     = covariance_weight_one*(measurement_measurement_covariance_matrix_one @ measurement_measurement_covariance_matrix_one.transpose()) + \
                                                    covariance_weight_two*(measurement_measurement_covariance_matrix_two @ measurement_measurement_covariance_matrix_two.transpose()) + \
                                                    measurement_covariance_matrix    
    
    # Enforce Covariance Symmetry
    measurement_measurement_covariance_matrix = 0.5*(measurement_measurement_covariance_matrix + measurement_measurement_covariance_matrix.transpose())

    # State-Measurement Covariance Matrix
    state_measurement_covariance_scale  = linalg.cholesky(covariance_weight_one * predicted_estimation_error_covariance_matrix, lower=True)
    state_measurement_covariance_matrix = state_measurement_covariance_scale @ (sigma_points_measurement[:, 1:number_of_states+1] - sigma_points_measurement[:, number_of_states+1:number_of_sigma_points]).transpose()

    # Kalman Gain
    kalman_gain_matrix = state_measurement_covariance_matrix @ linalg.inv(measurement_measurement_covariance_matrix)

    # Linear Minimum Mean Squared Error Estimate
    corrected_state_vector  = predicted_state_vector + kalman_gain_matrix @ (pseudorange_measurement - estimate_pseudorange_measurement)
    
    # Linear Minimum Mean Squared Error Covariance
    corrected_estimation_error_covariance = predicted_estimation_error_covariance_matrix - kalman_gain_matrix @ measurement_measurement_covariance_matrix @ kalman_gain_matrix.transpose()
    
    # Enforce Covariance Symmetry
    corrected_estimation_error_covariance = 0.5*(corrected_estimation_error_covariance + corrected_estimation_error_covariance.transpose())     
    
    return corrected_state_vector, corrected_estimation_error_covariance