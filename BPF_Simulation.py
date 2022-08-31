#  ============================================================================
#  Name        : BPF_Simulation.py
#  Description : Two-dimensional (2-D) bootstrap particle filter (BPF) 
#                implemented on a vehicle performing opportunistic navigation 
#                in a global navigation satellite system (GNSS)-denied 
#                environment using terrestrial signals of opportunity (SoOP).   
#  Author      : Alex Nguyen
#  Date        : August 2022
#  ============================================================================
#
#  If you have any questions or comments about this code, please do not hesitate
#  to send me an email at "alexaan2@uci.edu".
#
#  ============================================================================

# Import Packages 
from Functions import Initialize_Nonlinear_Filters
from Functions import Bootstrap_Particle_Filter
import timeit
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import linalg

# ======================================================================== TUNABLE PARAMETERS ============================================================================= #
''' FYI, you may tune the parameters below such that the desired environment settings are obtained. '''

# Number of SoOPs
m  = 0                                                                                        # Unknown SoOPs
n  = 10                                                                                       # Partially-Known (Position States) SoOPs 

# Simulation Time
T     = 0.01                                                                                  # Sampling Period [s]
t_end = 100                                                                                   # End Time [s]

# State Vector 
x_rx0 = np.array([0, 50, 15, -5, 10, 1]).reshape(6, 1)                                        # Receiver State Vector    

# Clock Quality [e.g., 'CSAC', 'Best OCXO', 'Typical OCXO', 'Typical TCXO', 'Worst TCXO']
h0_rx, hneg2_rx = Initialize_Nonlinear_Filters.clockQuality('Typical OCXO')                   # Receiver's Crystal Oscillator Quality
h0_s, hneg2_s   = Initialize_Nonlinear_Filters.clockQuality('Typical OCXO')                   # SoOPs' Crystal Oscillator Quality 

# Reciever Dynamics Power Spectral Density (PSD)
qx = 0.25                                                                                     # East Position Process Noise
qy = qx                                                                                       # North Position Process Noise

# Measurement Noise
measurement_noise = 25                                                                        

# Initial Estimation Error Covariance Matrices
P_rx0  = linalg.block_diag(2**2*np.eye(2), 1**2*np.eye(2), 30**2, 0.3**2)                     # Receiver States
P_s0   = linalg.block_diag(1e3*np.eye(2), 30**2, 0.3**2)                                      # Unknown SoOP States (IF ANY)
P_clk0 = linalg.block_diag(30**2, 0.3**2)                                                     # Partially-Known SoOP States

"""
Initial estimation error covariance matrices used for the KF variants filters which WILL NOT WORK for the BPF!
Using these settings will yield a diverging filter.

P_rx0  = linalg.block_diag(10**2*np.eye(2), 5**2*np.eye(2), 300**2, 3**2)                     # Receiver States
P_s0   = linalg.block_diag(1e3*np.eye(2), 300**2, 3**2)                                       # Unknown SoOP States (IF ANY)
P_clk0 = linalg.block_diag(300**2, 3**2)                                                      # Partially-Known SoOP States
"""

# Particle Filter Parameters
N             = 1*10**3                                                                       # Number of Particles
w             = np.ones((N, 1))/N                                                             # Particle Weights
Nthr          = 2*N/3                                                                         # Effective Particle Resampling Threshold
resample_type = 'Systematic Sampling'                                                         # Resampling Type ['Systematic Sampling' or 'Stratified Sampling']

# 1, 2, or 3-Sigma (68%, 95%, or 99.7%) Confidence Intervals 
sigma_bound = 3         

# ========================================================================================================================================================================== #

""" Simulation Parameters """
# Number of Measurements and States
nz = n + m                                                                                    # Total Number of SoOP Measurements
nx = 4 + 2*n + 4*m                                                                            # Total Number of States
 
# Time
c     = 299792458                                                                             # Speed of Light [m/s]
t     = np.arange(0, t_end + T, T)                                                            # Experiment Time Duration [s]
simulation_length = np.size(t)                                                                # Simulation Time Length [samples]

# Initialize SoOP Parameters
x_s0 = Initialize_Nonlinear_Filters.initializeSoopVector(nz)                                  # State Vector             

""" Power Spectral Density """
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
Qclk_s = np.array([[S_wts*T + S_wtsdot*T**3/3, S_wtsdot*T**2/2], 
                   [S_wtsdot*T**2/2, S_wtsdot*T]])

""" Receiver Dynamics """
# Velocity Random Walk State Transition Matrix
Fpv = np.block([[np.eye(2), np.eye(2)*T], 
                [np.zeros((2, 2)), np.eye(2)]])

# Receiver State Transition Matrix
Fr = linalg.block_diag(Fpv, Fclk)

# Receiver Process Covariance
Qpv    = np.array([[qx*T**3/3, 0, qx*T**2/2, 0], [0, qy*T**3/3, 0, qy*T**2/2], 
                   [qx*T**2/2, 0, qx*T, 0], [0, qy*T**2/2, 0, qy*T]])
Qclk_r = np.array([[S_wtr*T + S_wtrdot*T**3/3, S_wtrdot*T**2/2], 
                   [S_wtrdot*T**2/2, S_wtrdot*T]])

""" Initialize Nonlinear Filter"""
# Construct Necessary Matrices (e.g., T, F, G, P, Q, R)
LT, F, G, P, Q, R = Initialize_Nonlinear_Filters.matrixInitialization(Fpv, Fs, Fclk, Qpv, Qclk_r, Qclk_s, P_rx0, P_s0, P_clk0, measurement_noise, n, m)

# Process and Measurement Noise
q = linalg.cholesky(Q, lower=True)
r = linalg.cholesky(R, lower=True) 

# Construct Bootstrap Particle Filter State Vector
P_est = P
x_true, x_est, u = Initialize_Nonlinear_Filters.constructStateVector(n, m, x_rx0, x_s0, P_est, LT)
np.random.seed(1)
particles = np.random.multivariate_normal(x_true.reshape(nx, ), P_est, size=N).transpose()

""" Bootstrap Particle Filter """
# Preallocation
x_true_hist = np.zeros((nx, simulation_length))
x_est_hist  = np.zeros((nx, simulation_length))
P_std_hist  = np.zeros((nx, simulation_length))
zk_hist     = np.zeros((nz, simulation_length))
zk_hat_hist = np.zeros((nz, simulation_length)) 

# Start Timer
start = timeit.default_timer()    

for k in range(simulation_length):       
    # Set Random Noise Seed
    np.random.seed(k)
    wk = q @ np.random.randn(nx, 1)  
    np.random.seed(k)                                            
    vk = r @ np.random.randn(nz, 1)
    
    # True Pseudorange Measurements 
    h_zk = Initialize_Nonlinear_Filters.truePseudorangeMeasurements(x_true, x_s0, n, m)
    zk   = h_zk + vk
    
    # Propogate Particles
    particles_propogated = Bootstrap_Particle_Filter.propogateParticles(k, particles, N, nx, F, Q)
    
    # Estimate Pseudorange Measurements
    particle_measurement, zk_hat = Bootstrap_Particle_Filter.estimatedPseudorangeMeasurements(n, m, nx, N, particles_propogated, w, x_s0)
    
    # Compute Likelihood of Particles
    w = Bootstrap_Particle_Filter.computeLikelihood(nz, N, zk, particle_measurement, w, R)
    
    # Resample Particles (If Necessary)
    Neff = 1/np.sum(w**2)
    
    if Neff < Nthr:
        # Update Particles
        particles, w = Bootstrap_Particle_Filter.particleResampling(resample_type, nx, N, particles_propogated, w)
        
    else:
        # Particles Remain Unchanged
        particles = particles_propogated
        
    # Estimation Statistics
    x_correct, P_correct = Bootstrap_Particle_Filter.estimationStatistics(N, particles, w)    
        
    # Save Values
    x_true_hist[:, k:k+1] = x_true
    x_est_hist[:, k:k+1]  = x_correct
    P_std_hist[:, k:k+1]  = np.sqrt(np.diag(P_correct)).reshape(nx, 1)
    zk_hist[:, k:k+1]     = zk
    zk_hat_hist[:, k:k+1] = zk_hat
    
    # Update State Space 
    x_true = F @ x_true + G @ u + wk
    x_est  = x_correct
    P_est  = P_correct

# Distance Traveled
total_distance = np.sum(np.sqrt((np.diff(x_true_hist[0, :])**2 + np.diff(x_true_hist[1, :])**2)))
    
# End Timer
end = timeit.default_timer()                   
print("\nEnvironment:", n, "partially-known SoOPs and", m, "unknown SoOPs")   
print("Total Distance Traveled =", '%.2f' % total_distance, "m over", t[-1], "secs\n")                                        
print("BPF elapsed time =", '%.4f' % (end - start), "seconds")

# Estimation Error Trajectories
x_tilde_hist = x_true_hist - x_est_hist

""" Simulation Results """
# Simulation Layout Plots
plt.figure()
plt.plot(x_true_hist[0, :], x_true_hist[1, :], 'g', linewidth=2)
plt.plot(x_est_hist[0, :], x_est_hist[1, :], 'b', linewidth=2)
plt.scatter(x_s0[0, :], x_s0[1, :], s=15, c='r')
if m > 0:
    # Unknown SoOP Index
    unknown_index = (4 + 2*n + 1) - 1
    
    for i_m in range(m):
        # Plot Unknown SoOPs
        plt.scatter(x_est_hist[unknown_index, -1], x_est_hist[unknown_index+1, -1], s=10, c='k', marker="*")
        
        # Update Index
        unknown_index += 4
plt.xlabel('East (m)') 
plt.ylabel('North (m)')
plt.legend(['Ground-Truth', 'Estimated', 'SoOP Locations', 'Estimated SoOP Locations'], loc='best')     
plt.title('Simulation Layout') 

# Estimated Error Trajectories Plots     
sigma_bound_text = str(sigma_bound)

plt.figure()
ylabel_one = ['East Error (m)', 'North Error (m)']

for i_fig in range(2):                                                                               # Position States
    # Plot
    plt.subplot(2, 1, i_fig + 1)
    plt.plot(t, x_tilde_hist[i_fig,:])
    plt.plot(t, sigma_bound*P_std_hist[i_fig,:],'--r')
    plt.plot(t, -sigma_bound*P_std_hist[i_fig,:],'--r')
    plt.ylabel(ylabel_one[i_fig])
    plt.xlim([t[0], t[-1]])       
    plt.minorticks_on()
    plt.grid(True, which='both')     
    
    # Add Labels
    if i_fig == 0:
        plt.title('Position States')
        plt.legend([r'$\tilde{x}_{\mathrm{east}}$', r'$\pm {} \sigma$'.format(sigma_bound)], loc='best')     
    
    elif i_fig == 1:
        plt.xlabel('Time (s)')
        plt.legend([r'$\tilde{x}_{\mathrm{north}}$', r'$\pm {} \sigma$'.format(sigma_bound)], loc='best')  

plt.figure()
ylabel_two = ['East Error (m/s)', 'North Error (m/s)']

for i_fig in range(2, 4):                                                                            # Velocity States
    # Plot
    plt.subplot(2, 1, i_fig - 1)
    plt.plot(t, x_tilde_hist[i_fig,:])
    plt.plot(t, sigma_bound*P_std_hist[i_fig,:],'--r')
    plt.plot(t, -sigma_bound*P_std_hist[i_fig,:],'--r')
    plt.ylabel(ylabel_two[i_fig - 2])
    plt.xlim([t[0], t[-1]])       
    plt.minorticks_on()
    plt.grid(True, which='both')   
    
    # Add labels
    if i_fig == 2:
        plt.title('Velocity States')
        plt.legend([r'$\tilde{\dot{x}}_{\mathrm{east}}$', r'$\pm {} \sigma$'.format(sigma_bound)], loc='best')  
    
    elif i_fig == 3:
        plt.xlabel('Time (s)')
        plt.legend([r'$\tilde{\dot{x}}_{\mathrm{north}}$', r'$\pm {} \sigma$'.format(sigma_bound)], loc='best')        

""" Navigation Solution Performance Metrics """    
# Root Mean Square Error (RMSE)
position_rmse = np.sqrt(np.mean(np.sum(x_tilde_hist[0:2,:]**2, axis=0)))
velocity_rmse = np.sqrt(np.mean(np.sum(x_tilde_hist[2:4,:]**2, axis=0)))

# Max Error
max_position_error = max(np.sqrt(np.sum(x_tilde_hist[0:2,:]**2, axis=0)))
max_velocity_error = max(np.sqrt(np.sum(x_tilde_hist[2:4,:]**2, axis=0)))

print("\t     RMSE: Position =", '%.4f' % position_rmse, "m & Velocity =", '%.4f' % velocity_rmse, "m/s")
print("\tMax error: Position =", '%.4f' % max_position_error, "m & Velocity =", '%.4f' % max_velocity_error, "m/s")

# Unknown SoOP (IF ANY)
if m > 0:
    # Unknown SoOP Index
    unknown_index = (4 + 2*n + 1) - 1
    
    # Initial and Final Unknown SoOP Errors
    initial_error = []
    final_error   = []
    
    for i_m in range(m):
        # Compute Initial and Final Errors 
        initial_error = np.append(initial_error, round(linalg.norm(x_tilde_hist[unknown_index:unknown_index+2, 0]), 4))
        final_error   = np.append(final_error, round(linalg.norm(x_tilde_hist[unknown_index:unknown_index+2, -1]), 4))
    
        # Update Index
        unknown_index += 4
        
    print("Unknown SoOP Towers:\t", *range(1, m + 1))
    print("\tInitial Error =", initial_error)
    print("\t  Final Error =", final_error)

""" Nonlinear Filter Parameters """
print("where N = {} (number of particles) and N_threshold = {} (resampling threshold)".format(N, Nthr))
print("\n")

# Show Plots
plt.show()  