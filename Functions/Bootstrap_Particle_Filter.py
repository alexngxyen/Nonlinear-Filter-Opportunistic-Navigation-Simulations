#  ============================================================================
#  Name        : Bootstrap_Particle_Filter.py
#  Description : Helper functions needed to run the 'BPF_Simulation.py' code  
#  Author      : Alex Nguyen
#  Date        : August 2022
#  ============================================================================

# Import Packages
import numpy as np
from scipy import linalg

# Functions
def propogateParticles(time_step, particles, number_of_particles, number_of_states, state_transition_matrix, process_covariance_matrix):
    """
    This function will propogate the particles through the state transition model for the bootstrap particle filter..
    """

    # Draw Process Noise for Particles
    np.random.seed(time_step)
    process_noise_particles = np.random.multivariate_normal(np.zeros((number_of_states, )), process_covariance_matrix, size=number_of_particles).transpose()

    # Propogate Particles Through Dynamics Model
    particles_propogated = state_transition_matrix @ particles + process_noise_particles

    return particles_propogated

def estimatedPseudorangeMeasurements(partially_known_soops, unknown_soops, number_of_states, number_of_particles, particles_propogated, weights, soop_state_vector):
    """
    This function will compute the pseudorange measurements at a specific time instance for the bootstrap particle filter.
    """        
    
    # Initialize
    number_of_measurements = partially_known_soops + unknown_soops
    partially_known_index  = (4 + 1) - 1 
    unknown_index          = (4 + 2*partially_known_soops + 1) - 1
    
    # Preallocate
    particle_measurement = np.zeros((number_of_measurements, number_of_particles))

    # Partially-Known SoOPs
    if partially_known_soops != 0:
        position_difference_known = particles_propogated[:2, :].reshape(2, 1, number_of_particles) - soop_state_vector[:2, :partially_known_soops].reshape(2, partially_known_soops, 1)
        distance_partially_known  = linalg.norm(position_difference_known, axis=0).reshape(partially_known_soops, number_of_particles)
        particle_measurement[:partially_known_soops, :] = distance_partially_known + particles_propogated[partially_known_index:unknown_index:2, :]
    
    # Unknown SoOPs
    if unknown_soops != 0:
        unknown_soop_positions  = np.vstack((particles_propogated[unknown_index:number_of_states:4,:].reshape(1, unknown_soops, number_of_particles), 
                                             particles_propogated[unknown_index+1:number_of_states+1:4,:].reshape(1, unknown_soops, number_of_particles)))
        position_difference_unknown = particles_propogated[:2, :].reshape(2, 1, number_of_particles) - unknown_soop_positions
        distance_unknown            = linalg.norm(position_difference_unknown, axis=0).reshape(unknown_soops, number_of_particles)
        particle_measurement[partially_known_soops:partially_known_soops+unknown_soops, :] = distance_unknown + particles_propogated[unknown_index+2:number_of_states+2:4, :]
         
    # Estimate Pseudorange Measurement
    estimate_pseudorange_measurement = particle_measurement @ weights
    
    return particle_measurement, estimate_pseudorange_measurement

def computeLikelihood(number_of_measurements, number_of_particles, true_pseudorange_measurement, particle_measurement, weights, measurement_covariance_matrix):
    """
    This function will compute the likelihood of each particle for the bootstrap particle filter.
    """
    
    # Compute Likelihood of Particles
    measurement_difference  = (true_pseudorange_measurement.reshape(number_of_measurements, 1, 1) - np.reshape(particle_measurement, (number_of_measurements, 1, number_of_particles)))
    exponential_not_reshape = np.exp(-0.5*measurement_difference**2/np.diag(measurement_covariance_matrix).reshape(number_of_measurements, 1, 1))
    exponential_weights     = np.reshape(exponential_not_reshape, (number_of_measurements, number_of_particles))
    
    # Log-Likelihood 
    log_exponential_weights             = np.log(np.prod(exponential_weights + np.finfo(float).eps, axis=0).transpose()).reshape(number_of_particles, 1) + np.log(weights)
    log_exponential_weights_compensated = np.exp(log_exponential_weights - np.max(log_exponential_weights))
    
    # Update Weights 
    weights_updated = log_exponential_weights_compensated / np.sum(log_exponential_weights_compensated)

    return weights_updated

def particleResampling(resample_type, number_of_states, number_of_particles, particles_propogated, weights):
    """
    This function will perform particle resampling if the effective number of particles falls below a specified threshold for the bootstrap particle filter.
    """
    
    # Determine Random Number Generation Type
    if resample_type == ('Stratified Sampling'):
        # Stratified Sampling
        random_number = ((np.arange(0, number_of_particles)).reshape(number_of_particles, 1) + np.random.rand(number_of_particles, 1)) / number_of_particles
        
    elif resample_type == ('Systematic Sampling'):
        # Systematic Sampling Randomness
        random_number = ((np.arange(0, number_of_particles)).reshape(number_of_particles, 1) + np.random.rand(1)) / number_of_particles
        
    else:
        # Uniform Randomness
        random_number = np.random.rand(number_of_particles, 1)
    
    # Compute Cummulative Weights
    cummulative_weights            = np.cumsum(weights, dtype=float)
    cummulative_weights_normalized = cummulative_weights / cummulative_weights[-1]
    
    # Re-Index Particles
    index_arguement = np.vstack((random_number.reshape(number_of_particles, 1), cummulative_weights_normalized.reshape(number_of_particles, 1)))
    index_one       = np.argsort(index_arguement.reshape(len(index_arguement), ))
    index_two       = np.argwhere(index_one < number_of_particles)
    index           = index_two - np.arange(0, number_of_particles).reshape(number_of_particles, 1)
    
    # Resampled Particles
    particles = particles_propogated[:, index].reshape(number_of_states, number_of_particles)
    
    # Reset Weights
    weights = np.ones((number_of_particles, 1)) / number_of_particles
    
    return particles, weights

def estimationStatistics(number_of_particles, particles, weights):
    """
    This function will compute the estimaton statistics for the bootstrap particle filter.
    """

    # State Estimate
    corrected_state_vector = particles @ weights
    
    # Estimation Error Covariance Matrix
    corrected_estimation_error_covariance = (particles - np.tile(corrected_state_vector, (1, number_of_particles))) @ np.diag(weights.reshape(number_of_particles,)) @ (particles - np.tile(corrected_state_vector, (1, number_of_particles))).transpose() 
    
    # Enforce Covariance Symmetry
    corrected_estimation_error_covariance = 0.5*(corrected_estimation_error_covariance + corrected_estimation_error_covariance.transpose())    

    return corrected_state_vector, corrected_estimation_error_covariance