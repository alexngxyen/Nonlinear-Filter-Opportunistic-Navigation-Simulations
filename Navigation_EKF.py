# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import random as rng
import math
import scipy.linalg

# Define Functions
## DO THIS IN ANOTHER SCRIPT??

# Initialize Simulation Parameters
def kalman_filter(x, P, n, npts, y, F, Q, H, R):
    """ Script for Kalman Filter for Linear State Space Model 

    Inputs:
        x: initial state estimate
        P: initial state covariance matrix
    Outputs:
        x: updated state estimate
        P: updated state covariance matrix
    """
    # Initialize
    x_hat = x
    P_hat = P
    x_hat_store = np.zeros((n,npts))
    P_hat_store = np.zeros((n,npts))
    # Kalman Filter Loop
    for i in range(npts):
        # Predict
        x_hat_pred = F @ x_hat
        P_hat_pred = F @ P_hat @ F.T + Q
        # Update
        K = P_hat_pred @ H.T @ np.linalg.inv(H @ P_hat_pred @ H.T + R)
        x_hat = x_hat_pred + K @ (y[:,i] - H @ x_hat_pred)
        P_hat = (np.eye(n) - K @ H) @ P_hat_pred
        # Store
        x_hat_store[:,i] = x_hat
        P_hat_store[:,i] = P_hat
    return x_hat_store, P_hat_store


# def kalman_filter(x, P):
#     # Initialize
#     n = x.shape[0]
#     x_p = np.zeros((n,1))
#     P_p = np.zeros((n,n))
#     # Prediction
#     x_p = F @ x
#     P_p = F @ P @ F.T + Q
#     # Update
#     K = P_p @ H.T @ np.linalg.inv(H @ P_p @ H.T + R)
#     x = x_p + K @ (y - H @ x_p)
#     P = P_p - K @ H @ P_p
#     return x, P