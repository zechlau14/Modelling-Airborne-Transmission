import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.signal import convolve

#PDE parameters
v = 0.15 # 0.15m/s or 0.8m/s
N_th = 150000 #200000

R = 5 #0.5

eval = 3 #20 for scenario 3, R 0.25

#Domain
l = 8 #7
b = 8 #3.5
h = 3
V = l * b * h 
Q = 6/3600 #6/3600
K = Q * V * (2*V)**(-1/3)

#source's position (currently the centre)
x_o = 4 #5
y_o = 4 #0.25

x = 8 #6.75 and 0.75
y = 7 #1 and 2.75

t_input = 60*60*eval
delta_t = 0.3 #0.25
n_t = int(t_input/delta_t)
t = np.linspace(0.1,t_input+0.1,n_t)

#set up convolution
S = np.full(len(t), R)

nodes = int( v*3600/(2*l) * eval + 1)

C_y = np.exp(-((y-y_o)**2)/(4*K*t)) + np.exp(-((y+y_o)**2)/(4*K*t))
C_x = np.exp(-((x-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t)**2)/(4*K*t))
for n in range(1,nodes+1):
    C_y = C_y + np.exp(-((y-y_o - 2*n*l)**2)/(4*K*t)) + np.exp(-((y+y_o + 2*n*l)**2)/(4*K*t))
    C_y = C_y + np.exp(-((y-y_o + 2*n*l)**2)/(4*K*t)) + np.exp(-((y+y_o - 2*n*l)**2)/(4*K*t))
    C_x = C_x + np.exp(-((x-x_o -v*t - 2*n*b)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t + 2*n*b)**2)/(4*K*t))
    C_x = C_x + np.exp(-((x-x_o -v*t + 2*n*b)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t - 2*n*b)**2)/(4*K*t))
I = 1/(4*np.pi*K*t) * C_y * C_x * np.exp(-Q*t)

C = convolve(S,I,mode='valid') * delta_t
#C_l = C / 0.2 * 1000
C_l = C / 0.3 * 1000
viral_load = 0.0000017 * 500 * 16 / 30
V_c = C_l * viral_load

print("C = " + str(C) + "particles/m^2")
#print("C = " + str(V_c) + "virus copies/L")