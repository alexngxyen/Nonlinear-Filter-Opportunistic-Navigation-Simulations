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
n  = 2                                                                               # Partially-Known (Position States) SoOPs 
nz = n + m                                                                            # Total Number of SoOP Measurements
nx = 4 + 2*n + 4*m                                                                    # Total Number of States
 
# Time
c = 299792458                                                                         # Speed of Light [m/s]
T = 0.1                                                                               # Sampling Period [s]
t = np.arange(0, 100 + T, T)                                                          # Experiment Time Duration [s]
simulation_length = np.size(t)                                                        # Simulation Time Length [samples]

# Initialize Receiver Parameters
x_rx0 = np.array([0, 50, 15, -5, 10, 1]).reshape(6, 1)                                 # State Vector      
h0_rx, hneg2_rx = Initialize_Nonlinear_Filters.clockQuality('Typical OCXO')           # Typical Oven-Controlled Crystal Oscillator (OCXO)

# Initialize SoOP Parameters
x_s0 = Initialize_Nonlinear_Filters.initializeSoopVector(nz)                          # State Vector                          
h0_s, hneg2_s = Initialize_Nonlinear_Filters.clockQuality('Typical OCXO')             # Typical Oven-Controlled Crystal Oscillator (OCXO) 

""" Power Spectral Density """
# Dynamics 
qx = 1.00                                                                             # East Position Process Noise
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
P_rx0  = linalg.block_diag(5**2*np.eye(2), 1**2*np.eye(2), 300**2, 3**2)                 # Receiver's Initial Estimation Error Covariance Matrix
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
    
    # True Pseudorange Measurements 
    h_zk = Initialize_Nonlinear_Filters.truePseudorangeMeasurements(x_true, x_s0, n, m)
    zk   = h_zk + vk
    
    # Prediction Step
    x_predict, P_predict = Extended_Kalman_Filter.predictionStep(x_est, u, F, G, Q, P_est)

    # Estimate Pseudorange Measurements
    zk_hat, H = Extended_Kalman_Filter.estimatedPseudorangeMeasurements(x_predict, x_s0, n, m)

    # Correction Step
    x_correct, P_correct = Extended_Kalman_Filter.correctionStep(x_predict, P_predict, zk, zk_hat, H, R)    

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
    
# End Timer
end = timeit.default_timer()                                                              
print("Elapsed Time for EKF:", end - start, "seconds")

# Estimation Error Trajectories
x_tilde_hist = x_true_hist - x_est_hist

# Delete
RMSE = np.sqrt(np.mean(x_tilde_hist**2, axis=1))
pos_RMSE = linalg.norm(sum(RMSE[0:2]))
vel_RMSE = linalg.norm(sum(RMSE[2:4]))
print("Position RMSE", pos_RMSE)
print("Velocity RMSE", vel_RMSE)

plt.figure()
plt.plot(x_true_hist[0,:], x_true_hist[1,:])
plt.plot(x_est_hist[0,:], x_est_hist[1,:])
plt.scatter(x_s0[0,:],x_s0[1,:], color = 'red')

plt.figure()
for a in range(nz):
    plt.plot(t, zk_hist[a, :] - zk_hat_hist[a, :])

plt.show()