import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.integrate import trapz
from scipy.integrate import cumtrapz

#PDE parameters
v = 0.15
R = [5,2.5,0.5,0.25]
N_th = 150000
Q_value = [0.000033, 0.0002, 0.00083, 0.0017]
K_value = [0.00088, 0.0053, 0.022, 0.044] 

scenario = 0
R = R[0]
Q = Q_value[scenario]
K = K_value[scenario]
eval = 3 #hours

#mesh
l= 8
b= 8
delta_x = 0.05 #0.1
n_x = int(l / delta_x + 1)
x = np.linspace(0,l,n_x)
y = np.linspace(0,b,n_x) 
X,Y = np.meshgrid(x,y)
Infected = np.zeros_like(X)

#source's position (currently the centre)
x_o = l/2
y_o = b/2

t_end = 60*60*eval 
delta_t = 0.3
n_t = int(t_end/delta_t)+1
t = np.linspace(0.1,t_end+0.1,n_t)

#set up convolution
S = np.full(len(t), R)

m = int(34*eval) #nodes

for i in range(len(x)):
    for j in range(int((len(y)+1)/2)):
        C_y = np.exp(-((y[j]-y_o)**2)/(4*K*t)) + np.exp(-((y[j]+y_o)**2)/(4*K*t))
        C_x = np.exp(-((x[i]-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t)**2)/(4*K*t))
        for n in range(1,m+1):
            C_y = C_y + np.exp(-((y[j]-y_o - 2*n*l)**2)/(4*K*t)) + np.exp(-((y[j]+y_o + 2*n*l)**2)/(4*K*t))
            C_y = C_y + np.exp(-((y[j]-y_o + 2*n*l)**2)/(4*K*t)) + np.exp(-((y[j]+y_o - 2*n*l)**2)/(4*K*t))
            C_x = C_x + np.exp(-((x[i]-x_o -v*t - 2*n*b)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t + 2*n*b)**2)/(4*K*t))
            C_x = C_x + np.exp(-((x[i]-x_o -v*t + 2*n*b)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t - 2*n*b)**2)/(4*K*t))
        I = 1/(4*np.pi*K*t) * C_y * C_x * np.exp(-Q*t)
        C = convolve(S,I)[0:len(t)]*delta_t
        C_int = trapz(C,t)
        if C_int > N_th: 
            Infected[j][i] = 1
            Infected[int(len(y)-1-j)][i] = 1 

Counter = 0
for i in range(len(x)):
    for j in range(len(y)):
        if Infected[j][i] > 0: Counter = Counter + 1

Prob = Counter / (len(x)*len(y)) * 100

print("Infection Risk = " + str(Prob) + "%")
