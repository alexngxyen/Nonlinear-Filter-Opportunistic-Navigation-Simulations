#  ============================================================================
#  Name        : CKF_Simulation.py
#  Description : Two-dimensional (2-D) cubature Kalman filter (CKF) implemented 
#                on a vehicle performing opportunistic navigation in a 
#                global navigation satellite system (GNSS)-denied environment
#                using terrestrial signals of opportunity (SoOP).   
#  Author      : Alex Nguyen
#  Date        : August 2022
#  ============================================================================
#  Notes       : - Tunable Parameters
#                    (i) Simulation Setup - 'n', 'm', 'T', 't_end', 'x_rx0', 
#                        'h0_rx', 'hneg2_rx', 'h0_s', 'hneg2_s','qx', 'qy', 
#                        'measurement_noise', 'P_rx0', 'P_s0', 'P_clk0',  
#                         'sigma_bound', and set random seeds.
#                    (ii) Nonlinear Filter - ...
#                - If you have any questions or comments about this code, please 
#                  message the following email address "alexaan2@uci.edu".
#  ============================================================================

# Import Packages
from Functions import Initialize_Nonlinear_Filters
from Functions import Extended_Kalman_Filter
import timeit
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import linalg

""" Simulation Parameters """
# SoOPs
m  = 0                                                                                # Unknown SoOPs
n  = 9                                                                                # Partially-Known (Position States) SoOPs 
nz = n + m                                                                            # Total Number of SoOP Measurements
nx = 4 + 2*n + 4*m                                                                    # Total Number of States
 
# Time
c     = 299792458                                                                     # Speed of Light [m/s]
T     = 0.1                                                                           # Sampling Period [s]
t_end = 100                                                                           # Simulation End Time [s]
t     = np.arange(0, t_end + T, T)                                                    # Experiment Time Duration [s]
simulation_length = np.size(t)                                                        # Simulation Time Length [samples]

# Initialize Receiver Parameters
x_rx0 = np.array([0, 50, 15, -5, 10, 1]).reshape(6, 1)                                # State Vector      
h0_rx, hneg2_rx = Initialize_Nonlinear_Filters.clockQuality('Typical OCXO')           # Typical Oven-Controlled Crystal Oscillator (OCXO)

# Initialize SoOP Parameters
x_s0 = Initialize_Nonlinear_Filters.initializeSoopVector(nz)                          # State Vector             
h0_s, hneg2_s = Initialize_Nonlinear_Filters.clockQuality('Typical OCXO')             # Typical Oven-Controlled Crystal Oscillator (OCXO) 

""" Power Spectral Density """
# Dynamics 
qx = 0.25                                                                             # East Position Process Noise
qy = qx                                                                               # North Position Process Noise

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
# Nonlinear Filter Parameters
measurement_noise = 25                            
P_rx0  = linalg.block_diag(10**2*np.eye(2), 5**2*np.eye(2), 300**2, 3**2)                # Receiver's Initial Estimation Error Covariance Matrix
P_s0   = linalg.block_diag(1e3*np.eye(2), 300**2, 3**2)                                  # Unknown SoOP's Initial Estimation Error Covariance Matrix
P_clk0 = linalg.block_diag(300**2, 3**2)                                                 # Partially-Known SoOP's Initial Estimation Error Covariance Matrix

# Construct Necessary Matrices (e.g., T, F, G, P, Q, R)
LT, F, G, P, Q, R = Initialize_Nonlinear_Filters.matrixInitialization(Fpv, Fs, Fclk, Qpv, Qclk_r, Qclk_s, P_rx0, P_s0, P_clk0, measurement_noise, n, m)

# Process and Measurement Noise
q = linalg.cholesky(Q, lower=True)
r = linalg.cholesky(R, lower=True) 

# Construct Extended Kalman Filter State Vector
P_est = P
x_true, x_est, u = Initialize_Nonlinear_Filters.constructStateVector(n, m, x_rx0, x_s0, P_est, LT)

""" Extended Kalman Filter """
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
    
    """INCLUDE NEW CODE""" 

    # Save Values
    x_true_hist[:, k:k+1] = x_true
    x_est_hist[:, k:k+1]  = x_correct
    P_std_hist[:, k:k+1]  = np.sqrt(P_correct.diagonal()).reshape(nx, 1)
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
print("EKF elapsed time =", '%.4f' % (end - start), "seconds")

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
sigma_bound = 3                                                                                      # 1, 2, or 3-sigma (68, 95, or 99.7) confidence interval    
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
        plt.legend([r'$\tilde{x}_{\mathrm{east}}$', r'$\pm 3 \sigma$'], loc='best')     
    
    elif i_fig == 1:
        plt.xlabel('Time (s)')
        plt.legend([r'$\tilde{x}_{\mathrm{north}}$', r'$\pm 3 \sigma$'], loc='best')  

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
        plt.legend([r'$\dot{\tilde{x}}_{\mathrm{east}}$', r'$\pm 3\sigma$'], loc='best')  
    
    elif i_fig == 3:
        plt.xlabel('Time (s)')
        plt.legend([r'$\dot{\tilde{x}}_{\mathrm{north}}$', r'$\pm 3 \sigma$'], loc='best')  

""" Navigation Solution Performance Metrics"""    
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
        initial_error = np.append(initial_error, linalg.norm(x_tilde_hist[unknown_index:unknown_index+2, 0]))
        final_error   = np.append(final_error, linalg.norm(x_tilde_hist[unknown_index:unknown_index+2, -1]))
    
        # Update Index
        unknown_index += 4
        
    print("Unknown SoOP Towers", *range(1, m + 1), ":")
    print("\tInitial Error =", '%.4f' % initial_error)
    print("\t  Final Error =", '%.4f' % final_error)

# Show Plots
plt.show()
print("\n")