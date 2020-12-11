import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.signal import convolve

#PDE parameters
v = 0.15 # 0.15m/s or 0.8m/s
R = [5,2.5,0.5,0.25]
N_th = 150000 #200000
Q_value = [0.000033, 0.0002, 0.000833, 0.00167]
K_value = [0.00088, 0.0053, 0.0220, 0.0441] 

R = R[3]
eval = 21 #20 for scenario 3, R 0.25
scenario = 3
Q = Q_value[scenario]
K = K_value[scenario]

#Q = 0.01/3600
#K = Q * 192 * (384**(-1/3))

#Domain
l = 8
b = 8

#source's position (currently the centre)
x_o = 4
y_o = 4

x = 8
y = 0

t_input = 60*60*eval
delta_t = 0.3 #0.25
n_t = int(t_input/delta_t)
t = np.linspace(0.1,t_input+0.1,n_t)

#set up convolution
S = np.full(len(t), R)

nodes = int(34*eval) 

C_y = np.exp(-((y-y_o)**2)/(4*K*t)) + np.exp(-((y+y_o)**2)/(4*K*t))
C_x = np.exp(-((x-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t)**2)/(4*K*t))
for n in range(1,nodes+1):
    C_y = C_y + np.exp(-((y-y_o - 2*n*l)**2)/(4*K*t)) + np.exp(-((y+y_o + 2*n*l)**2)/(4*K*t))
    C_y = C_y + np.exp(-((y-y_o + 2*n*l)**2)/(4*K*t)) + np.exp(-((y+y_o - 2*n*l)**2)/(4*K*t))
    C_x = C_x + np.exp(-((x-x_o -v*t - 2*n*b)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t + 2*n*b)**2)/(4*K*t))
    C_x = C_x + np.exp(-((x-x_o -v*t + 2*n*b)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t - 2*n*b)**2)/(4*K*t))
I = 1/(4*np.pi*K*t) * C_y * C_x * np.exp(-Q*t)

C = convolve(S,I)[0:len(t)] * delta_t

#test
C_test = np.trapz(C,t) #scalar

if C_test > N_th:
    C_int = cumtrapz(C,t,initial=0) #array

    i = len(t) - 1
    while C_int[i] > N_th:
        i = i-1
    ans = t[i] / (60)
    print(ans)
else:
    print("C_int = " + str(C_test))
    print("No solution found, need longer evaluation time")