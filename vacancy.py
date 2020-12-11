import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

#PDE parameters
v = 0.15 
R = [5,2.5,0.5,0.25]
Q = [0.000033, 0.0002, 0.00083, 0.0017]
K = [0.00088, 0.0053, 0.022, 0.044] 
#ss = [ [16,25,37,52] , [24,39,61,96] , [64,110,240,447] , [100,176,462,885] ] #OTL
ss = [ [16,25,37,52] , [24,39,61,96] , [64,110,90,120] , [100,176,90,120] ]

act = int(input("Enter act (0-3): "))
scenario = int(input("Enter scenario (0-3): "))
R = R[act]
Q = Q[scenario]
K = K[scenario]
ss = ss[act][scenario]

x = 5
y = 4
hour = 8 #36
C_target = 13.8

#Domain
l = 8 
b = 8  

#source's position
x_o = 4
y_o = 4

#time
t_end = 60*60 * hour
delta_t = 0.3
n_t = int(t_end/delta_t) + 1
t = np.linspace(0.1,t_end+0.1,n_t)
n_ss = int(ss*60 / delta_t)

#set up convolution
S = np.zeros_like(t)
S[0:n_ss] = np.full(n_ss, R)

k = int(34*hour) #nodes 34 for 1h

C_y = np.exp(-((y-y_o)**2)/(4*K*t)) + np.exp(-((y+y_o)**2)/(4*K*t))
C_x = np.exp(-((x-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t)**2)/(4*K*t))
for n in range(1,k+1):
    C_y = C_y + np.exp(-((y-y_o - 2*n*l)**2)/(4*K*t)) + np.exp(-((y+y_o + 2*n*l)**2)/(4*K*t))
    C_y = C_y + np.exp(-((y-y_o + 2*n*l)**2)/(4*K*t)) + np.exp(-((y+y_o - 2*n*l)**2)/(4*K*t))
    C_x = C_x + np.exp(-((x-x_o -v*t - 2*n*b)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t + 2*n*b)**2)/(4*K*t))
    C_x = C_x + np.exp(-((x-x_o -v*t + 2*n*b)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t - 2*n*b)**2)/(4*K*t))
I = 1/(4*np.pi*K*t) * C_y * C_x * np.exp(-Q*t)
C = convolve(S,I)[0:len(t)] * delta_t 

t = t / 60

if C[-1] > C_target: print("Evaluation time is not enough to clean the air.")
else:
    n = n_ss
    while C[n] > C_target : n = n+1 
    t_vacant = t[n] - ss
    print("Required vacancy time is " + str(int(t_vacant)) + "minutes.")
