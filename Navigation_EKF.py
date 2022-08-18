# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import random as rng
import math
import scipy.linalg

""" Functions """
def initial_xs0(N_SoOPs):
    """
    Initialize the state vector for the SoOPs
    """
    # Random SoOP Locations
    rng.seed(1)
    x = (1000 - (-100))*np.random.rand(N_SoOPs, 1) + (-100)

    rng.seed(2)
    y = (500 - (-500))*np.random.rand(N_SoOPs, 1) + (-500)

    # Clock Error States
    t  = np.ones((N_SoOPs, 1))
    dt = 0.1 * np.ones((N_SoOPs, 1))

    # Construct SoOP State Vector
    xs0 = np.transpose(np.hstack((x, y, t, dt)))
    return xs0

def Clock_Quality(clk_type):
    """
    Define the clock quality for the SoOPs
    """
    if clk_type == ('CSAC'):
        h0 = 2.0e-20
        h2 = 4.0e-29   
    if clk_type == ('Best OCXO'):
        h0 = 2.6e-22
        h2 = 4.0e-26  
    if clk_type == ('Typical OCXO'):
        h0 = 8.0e-20
        h2 = 4.0e-23 
    if clk_type == ('Typical TCXO'):
        h0 = 9.4e-20
        h2 = 3.8e-21 
    if clk_type == ('Worst TCXO'):
        h0 = 2.0e-19
        h2 = 2.0e-20 
    return h0, h2

"""Simulation Parameters"""
# Number of SoOPs
m = 0                                                    # Fully Unknown SoOPs
n = 10                                                   # Partially-Known (Position States) SoOPs 
M = n + m                                                # Total Number of SoOPs
nx = 4 + 2*n + 4*m                                       # Total Number of States

# Time
c = 299792458                                            # Speed of Light [m/s]
T = 10e-2                                                # Sampling Period [s]
t = np.arange(0, 50 + T, T)                              # Experiment Time Duration [s]
SimL = np.size(t)                                        # Simulation Time Length

# Initialize Receiver Parameters
x_rx0 = np.array([0, 50, 15, -5, 100, 1]).reshape(6, 1)        
h0_rx, hneg2_rx = Clock_Quality('Typical OCXO')          # Typical Oven-Controlled Crystal Oscillator (OCXO)

# Initialize SoOP Parameters
x_s0 = initial_xs0(M)                                 
h0_s, hneg2_s = Clock_Quality('Best OCXO')               #  Best Oven-Controlled Crystal Oscillator (OCXO) 

""" Power Spectral Density """
# Dynamics Process Noise
qx = 0.25
qy = qx

# Receiver Process Noise
Sdt = h0_rx/2
Sddt = 2*math.pi**2*hneg2_rx

# SoOP Process Noise
S_wtr = h0_s/2; 
S_wtrdot = 2*math.pi**2*hneg2_s

""" SoOP Dynamics"""
# Clock Error State Transition Matrix
Fclk = np.array([[1, T], [0, 1]])

# SoOP State Transition Matrix
