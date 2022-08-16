# Import Packages
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

# Define Functions
def get_q_clk(clk_type,T):
    c = 299792458
    if clk_type == 'Typical TCXO':
        h0 = 9.4e-20
        h_2 = 3.8e-21
    if clk_type == 'Typical OCXO':
        h0 = 8.0e-20
        h_2 = 4.0e-23
    Sdt = h0/2
    Sddt = 2*math.pi**2*h_2
    q_clk = c**2*np.array([[Sdt*T+Sddt*T**3/3, Sddt*T**2/2],[Sddt*T**2/2, Sddt*T]])
    return q_clk
    
# Initialize   
npts = 850
T = 0.1
r_sop = np.array([[250,100],[0,-100],[500,0],[300,300],[700,-300],[1200,0]]).transpose()
N_sop = r_sop.shape[1]

F_clk = np.array([[1,T],[0,1]])
FF_clk = np.kron(np.eye(N_sop+1),F_clk)
F_clk = np.kron(np.eye(N_sop),F_clk)
F_pv = np.eye(4)
F_pv[0:2,2:4] = T*np.eye(2)
q_xy = 1*0.1
Q_pv = np.zeros((4,4))
Q_pv[0:2,0:2] = np.eye(2)*q_xy*T**3/3
Q_pv[0:2,2:4] = np.eye(2)*q_xy*T**2/2
Q_pv[2:4,0:2] = np.eye(2)*q_xy*T**2/2
Q_pv[2:4,2:4] = np.eye(2)*q_xy*T

P_clk0_sop = np.kron(np.eye(N_sop),np.diag([1e0*9e4,1e0*9e0]))
x_clk0_sop = np.random.multivariate_normal(np.zeros(2*N_sop),P_clk0_sop)
    
sop_clk_qual = 'Typical OCXO'
Q_clk_sop = get_q_clk(sop_clk_qual,T)

r0_rx = np.array([0,10])
v0_rx = np.array([10,0])
P_clk0_rx = np.diag([1e0*9e6,1e0*9e2])
x_clk0_rx = np.random.multivariate_normal(np.zeros(2),P_clk0_rx)

rx_clk_qual = 'Typical TCXO'
Q_clk_rx = get_q_clk(rx_clk_qual,T)

QQ_clk = np.zeros((2+2*N_sop,2+2*N_sop))
QQ_clk[0:2,0:2] = Q_clk_rx
QQ_clk[2:,2:] = np.kron(np.eye(N_sop),Q_clk_sop)

x_clk0 = np.concatenate((x_clk0_rx,x_clk0_sop))
x_clk = np.zeros((2+2*N_sop,npts))
x_clk[:,0] = x_clk0
x_true = np.zeros((4+2*N_sop,npts))
P_pv = np.diag([100,100,10,10])
x_true[0:4,0] = np.random.multivariate_normal(np.concatenate((r0_rx,v0_rx)),P_pv)
z = np.zeros((N_sop,npts))
R = 9
for k in range(npts):
    if k != npts-1:
        x_true[0:4,k+1] = np.dot(F_pv,x_true[0:4,k]) + np.random.multivariate_normal(np.zeros(4),Q_pv)
        x_clk[:,k+1] = np.dot(FF_clk,x_clk[:,k]) + np.random.multivariate_normal(np.zeros(2+2*N_sop),QQ_clk)
    for i in range(N_sop):
        z[i,k] = scipy.linalg.norm(x_true[0:2,k]-r_sop[0:2,i]) + x_clk[0,k] - x_clk[2+2*i,k] + np.random.normal(0,np.sqrt(R))

for i in range(N_sop):
    x_true[4+2*i:6+2*i,:] = x_clk[0:2,:] - x_clk[2+2*i:4+2*i,:]

Q_clk = np.kron(np.ones((N_sop,N_sop)),Q_clk_rx) + np.kron(np.eye(N_sop),Q_clk_sop)
P_clk = np.kron(np.ones((N_sop,N_sop)),P_clk0_rx) + P_clk0_sop
P_k1k1 = np.zeros((4+2*N_sop,4+2*N_sop))
P_k1k1[0:4,0:4] = P_pv
P_k1k1[4:,4:] = P_clk
F = scipy.linalg.block_diag(F_pv,F_clk)
Q = scipy.linalg.block_diag(Q_pv,Q_clk)

# Extended Kalman Filter (EKF)
x = np.zeros(x_true.shape)
x[0:4,0] = np.concatenate((r0_rx,v0_rx))
P = np.zeros(x_true.shape)
P[:,0] = np.diag(P_k1k1)

nu = np.zeros((N_sop,npts-1))
for k in range(npts-1):
    x[:,k+1] = np.dot(F,x[:,k])
    P_k1k = np.dot(np.dot(F,P_k1k1),F.transpose()) + Q
    
    H = np.zeros((N_sop,4+2*N_sop))
    z_hat = np.zeros(N_sop)
    for i in range(N_sop):
        dr = x[0:2,k+1] - r_sop[:,i]
        z_hat[i] = scipy.linalg.norm(dr) + x[4+2*i,k+1]
        H[i,0:2] = dr/scipy.linalg.norm(dr)
        H[i,4+2*i] = 1
    
    S = np.dot(np.dot(H,P_k1k),H.transpose()) + R*np.eye(N_sop)
    K = np.dot(np.dot(P_k1k,H.transpose()),np.linalg.inv(S))
    nu[:,k] = z[:,k+1] - z_hat
    x[:,k+1] = x[:,k+1] + np.dot(K,nu[:,k])
    A = np.eye(4+2*N_sop) - np.dot(K,H)
    P_k1k1 = np.dot(np.dot(A,P_k1k),A.transpose()) + np.dot(np.dot(K,R*np.eye(N_sop)),K.transpose())
    P[:,k+1] = np.diag(P_k1k1)

# Root Mean Square Error
t = np.linspace(0,(npts-1)*T,npts)
x_err = x_true - x
RMSE = np.sqrt(sum(x_err.transpose()**2)/npts)
pos_RMSE = scipy.linalg.norm(RMSE[0:2])
vel_RMSE = scipy.linalg.norm(RMSE[2:4])
print("Position RMSE", pos_RMSE)
print("Velocity RMSE", vel_RMSE)

# # Plot Noise
# ranges = np.zeros((N_sop,npts))
# for i in range(N_sop):
#     ranges[i,:] = np.sqrt(sum((x_true[0:2,:]-np.dot(r_sop[0:2,i].reshape((2,1)),np.ones((1,npts))))**2))
# p = 2
#plt.plot(ranges[p,:] + x_true[4+2*p,:])
#plt.plot(z[p,:],'--')
# plt.figure()
# plt.plot(ranges[p,:] + x_true[4+2*p,:] - z[p,:])

# Plot Estimation Error Trajectories
sig = 3

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, x_err[0,:])
plt.plot(t, sig*np.sqrt(P[0,:]),'--r')
plt.plot(t, -sig*np.sqrt(P[0,:]),'--r')
plt.ylabel('East')
plt.title('Position Estimation Error Trajectories')
plt.xlim([t[0], t[-1]])

plt.subplot(2, 1, 2)
plt.plot(t, x_err[1,:])
plt.plot(t, sig*np.sqrt(P[1,:]),'--r')
plt.plot(t, -sig*np.sqrt(P[1,:]),'--r')
plt.ylabel('North')
plt.xlim([t[0], t[-1]])

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, x_err[2,:])
plt.plot(t, sig*np.sqrt(P[2,:]),'--r')
plt.plot(t, -sig*np.sqrt(P[2,:]),'--r')
plt.ylabel('East')
plt.title('Velocity Estimation Error Trajectories')
plt.xlim([t[0], t[-1]])

plt.subplot(2, 1, 2)
plt.plot(t, x_err[3,:])
plt.plot(t, sig*np.sqrt(P[3,:]),'--r')
plt.plot(t, -sig*np.sqrt(P[3,:]),'--r')
plt.ylabel('North')
plt.xlim([t[0], t[-1]])

# Plot Environment Layout
plt.figure()
plt.plot(x_true[0,:], x_true[1,:])
plt.plot(x[0,:], x[1,:])
plt.scatter(r_sop[0,:],r_sop[1,:], color = 'red')

# Plot Innovation
plt.figure()
plt.title('Innovation')
plt.plot(t[1:], nu[1,:], color='green')
plt.ylim(-10,10)

# Show Plots
plt.show()