import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.integrate import cumtrapz

#parameters
time = 4 #event duration, in hours
l = 8 #x-length (m)
w = 8 #y-length (m)
h = 3 #z-length (m)
x_o = 4 #x-coordinate of source
y_o = 4 #y-coordinate of source
v = 0.15 #air velocity (m/s) from left to right. 
R = 5 #aerosol generation rate (particles/s)
x = 8 #x-coordinate evaluated
y = 8 #y-coordinate evaluated
p = 1.3*10**(-4) #breathing rate (m^3/s)
k = 0.0069 #infectivity constant for dose-response model
Q = 0.0002 # Air exchange rate (s^-1)
K = 0.0053 # Eddy diffusion coefficient (m^2/s)
d = 1.7*10**(-4) #deactivation rate (s^-1)
s = 1.1*10**(-4) #settling rate (s^-1)

#time-axis
t_end = 60*60*time #in seconds
delta_t = 1 #(s) time-steps
n_t = int(t_end/delta_t)
t = np.linspace(delta_t,t_end,n_t)

#initialize source function. multiply by delta_t to discretize the function.
S = delta_t * np.full(len(t), R)

#Impulse function I
#I_y is an approximation of the sum of y-exponentials in the impulse function
I_y = np.exp(-((y-y_o)**2)/(4*K*t)) + np.exp(-((y+y_o)**2)/(4*K*t))
for n in range(1,4):
    I_y += np.exp(-((y-y_o - 2*n*w)**2)/(4*K*t)) + np.exp(-((y+y_o + 2*n*w)**2)/(4*K*t))
    I_y += np.exp(-((y-y_o + 2*n*w)**2)/(4*K*t)) + np.exp(-((y+y_o - 2*n*w)**2)/(4*K*t))
#I_x is an approximation of the sum of x-exponentials in the impulse function
I_x = np.exp(-((x-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t)**2)/(4*K*t))
m = int(v*3600/(2*l) *time) #no of times the particles travel around the recirculation loop
for n in range(1,m+1):
    I_x += np.exp(-((x-x_o -v*t + 2*n*l)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t - 2*n*l)**2)/(4*K*t))
    I_x += np.exp(-((x-x_o -v*t - 2*n*l)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t + 2*n*l)**2)/(4*K*t))
sinks = np.exp(-(Q+d+s)*t) #effect of ventilation, deactivation and settling sinks
I = 1/(4*np.pi*K*t) * I_y * I_x * sinks #Impulse function

#Convolve S and I to find C
C = convolve(S,I)[0:len(t)] / (h/2)
dose = p * cumtrapz(C,t)
P = 1 - np.exp(-dose * k)

t = t/60 #convert time-axis to minutes for plotting of graph

P = P * 100 #convert probability to percent for plotting of graph
#plot Probability versus time
plt.plot(t[1:],P)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Time (min)")
plt.ylabel("Probability ($\%$)")
plt.show()
