import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

#PDE parameters
v = 0.15 # 0.15m/s
R = 1
Q_values = [0.000033, 0.0002, 0.000833, 0.00167]
K_values = [0.00088, 0.0053, 0.0220, 0.0441] 

hour = 24
scenario = 0
Q = Q_values[scenario]
K = K_values[scenario]

#Domain
l = 8 
b = 8  

#source's position (currently the centre)
x_o = 4
y_o = 4

x = 8
y = 8

# graph's axis
t_end = 60*60 * hour
delta_t = 0.3 # 0.025
n_t = int(t_end/delta_t) + 1
t = np.linspace(0.1,t_end+0.1,n_t)

#set up convolution
S = np.full(len(t), R)

k = int(34*hour) #nodes 34 for 1h

C_y = np.exp(-((y-y_o)**2)/(4*K*t)) + np.exp(-((y+y_o)**2)/(4*K*t))
C_x = np.exp(-((x-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t)**2)/(4*K*t))
for n in range(1,k+1):
    C_y = C_y + np.exp(-((y-y_o - 2*n*l)**2)/(4*K*t)) + np.exp(-((y+y_o + 2*n*l)**2)/(4*K*t))
    C_y = C_y + np.exp(-((y-y_o + 2*n*l)**2)/(4*K*t)) + np.exp(-((y+y_o - 2*n*l)**2)/(4*K*t))
    C_x = C_x + np.exp(-((x-x_o -v*t - 2*n*b)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t + 2*n*b)**2)/(4*K*t))
    C_x = C_x + np.exp(-((x-x_o -v*t + 2*n*b)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t - 2*n*b)**2)/(4*K*t))
I = 1/(4*np.pi*K*t) * C_y * C_x * np.exp(-Q*t)
C = convolve(S,I)[0:len(t)] * delta_t * 0.5

print("C_8h = " + str(C[-1])) 

delta_i = int(60/delta_t)
for i in range(0,len(t)-delta_i):
    if C[i+delta_i] - C[i] > 0.01: n_ss = i

t = t/60
print("t_ss = " + str(t[n_ss]) + "min")
print("C_ss = " + str(C[n_ss]))