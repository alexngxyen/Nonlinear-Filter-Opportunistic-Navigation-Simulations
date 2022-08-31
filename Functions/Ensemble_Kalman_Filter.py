#  ============================================================================
#  Name        : Ensemble_Kalman_Filter.py
#  Description : Helper functions needed to run the 'EnKF_Simulation.py' code  
#  Author      : Alex Nguyen
#  Date        : August 2022
#  ============================================================================

# Import Packages
import numpy as np
from scipy import linalg

# Functions
def predictionStep(time_step, number_of_states, number_of_ensembles, ensemble_members, state_transition_matrix, process_covariance_matrix):
    """
    This function will perform the time-update (prediction step) for the ensemble Kalman filter.
    """
    
    # Generate Noise for Ensemble Members
    np.random.seed(time_step)
    process_noise_ensemble_members = linalg.cholesky(process_covariance_matrix, lower=True) @ np.random.randn(number_of_states, number_of_ensembles)
    
    # Propogate Ensemble Members 
    ensemble_members_predict = state_transition_matrix @ ensemble_members + process_noise_ensemble_members
    
    # Compute Empirical Mean and Covariance
    predicted_state_vector = np.mean(ensemble_members_predict, axis=1).reshape(number_of_states, 1)
    predicted_estimation_error_covariance = (ensemble_members_predict - np.tile(predicted_state_vector, (1, number_of_ensembles))) @ (ensemble_members_predict - np.tile(predicted_state_vector, (1, number_of_ensembles))).transpose() / (number_of_ensembles - 1)
    
    # Enforce Covariance Symmetry
    predicted_estimation_error_covariance = 0.5*(predicted_estimation_error_covariance + predicted_estimation_error_covariance.transpose())                                             
  
    return predicted_state_vector, predicted_estimation_error_covariance, ensemble_members_predict

def estimatedPseudorangeMeasurements(partially_known_soops, unknown_soops, number_of_ensembles, ensemble_members_predict, soop_state_vector):
    """
    This function will compute the pseudorange measurements at a specific time instance for the ensemble Kalman filter.
    """
    
    # Initialize
    number_of_measurements = partially_known_soops + unknown_soops
    partially_known_index  = (4 + 1) - 1 
    unknown_index          = (4 + 2*partially_known_soops + 1) - 1
    
    # Preallocate
    ensemble_members_measurement = np.zeros((number_of_measurements, number_of_ensembles))
    
    for i in range(number_of_measurements):
        if i < partially_known_soops:
            # Compute Estimate Pseudorange Measurement
            distance                               = linalg.norm((ensemble_members_predict[:2, :] - soop_state_vector[:2, i:i+1]).reshape(2, number_of_ensembles), axis=0).reshape(1, number_of_ensembles)
            ensemble_members_measurement[i:i+1, :] = distance + ensemble_members_predict[partially_known_index:partially_known_index+1, :]
            
            # Update Index
            partially_known_index += 2
            
        elif i >= partially_known_soops and i <= number_of_measurements:
            # Compute Estimate Pseudorange Measurement
            distance                               = linalg.norm((ensemble_members_predict[:2, :] - ensemble_members_predict[unknown_index:unknown_index+2, :]).reshape(2, number_of_ensembles), axis=0).reshape(1, number_of_ensembles)
            ensemble_members_measurement[i:i+1, :] = distance + ensemble_members_predict[unknown_index+2:unknown_index+3, :]
                        
            # Update Index
            unknown_index += 4
            
    # Compute Estimate Pseudorange Measurement
    estimate_pseudorange_measurement = np.mean(ensemble_members_measurement, axis=1).reshape(number_of_measurements, 1)
    
    return estimate_pseudorange_measurement, ensemble_members_measurement      

def correctionStep(time_step, number_of_ensembles, number_of_measurements, number_of_states, ensemble_members_predict, true_pseudorange_measurement, estimate_pseudorange_measurement, 
                   ensemble_members_measurement, measurement_covariance_matrix, predicted_state_vector):
    """
    This function will perform the measurement-update (correction step) for the ensemble Kalman filter.
    """
    
    # Measurement-Measurement Covariance Matrix
    ensemble_members_minus_estimate_measurement = (ensemble_members_measurement - np.tile(estimate_pseudorange_measurement, (1, number_of_ensembles)))
    measurement_measurement_covariance_matrix   = ensemble_members_minus_estimate_measurement @ ensemble_members_minus_estimate_measurement.transpose()/(number_of_ensembles-1) + measurement_covariance_matrix
    
    # Enforce Covariance Symmetry
    measurement_measurement_covariance_matrix = 0.5*(measurement_measurement_covariance_matrix + measurement_measurement_covariance_matrix.transpose())

    # State-Measurement Covariance Matrix
    ensemble_members_minus_predicted_state_vector = (ensemble_members_predict - np.tile(predicted_state_vector, (1, number_of_ensembles)))
    state_measurement_covariance_matrix           = ensemble_members_minus_predicted_state_vector @ ensemble_members_minus_estimate_measurement.transpose()/(number_of_ensembles-1)

    # Kalman Gain
    kalman_gain_matrix = state_measurement_covariance_matrix @ linalg.inv(measurement_measurement_covariance_matrix)
    
    # Update Ensemble Members
    np.random.seed(time_step)
    measurement_noise_ensemble_members = linalg.cholesky(measurement_covariance_matrix, lower=True) @ np.random.randn(number_of_measurements, number_of_ensembles)
    ensemble_members = ensemble_members_predict + kalman_gain_matrix @ (np.tile(true_pseudorange_measurement, (1, number_of_ensembles)) - (ensemble_members_measurement + measurement_noise_ensemble_members))

    # Compute Empirical Mean and Covariance
    corrected_state_vector = np.mean(ensemble_members, axis=1).reshape(number_of_states, 1)
    corrected_estimation_error_covariance = (ensemble_members - np.tile(corrected_state_vector, (1, number_of_ensembles))) @ (ensemble_members - np.tile(corrected_state_vector, (1, number_of_ensembles))).transpose() / (number_of_ensembles - 1)
    
    # Enforce Covariance Symmetry
    corrected_estimation_error_covariance = 0.5*(corrected_estimation_error_covariance + corrected_estimation_error_covariance.transpose())       
    
    return corrected_state_vector, corrected_estimation_error_covariance, ensemble_members    