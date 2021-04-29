import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.integrate import cumtrapz

#parameters
time = 12 #24 #event duration, in hours
l = 8 #x-length (m)
w = 8 #y-length (m)
h = 3 #z-length (m)
x_o = 4 #x-coordinate of source
y_o = 4 #y-coordinate of source
R = 5 #aerosol generation rate (speaking)
v = 0.15 #air velocity (m/s) from left to right. 
x = 5 #x-coordinate evaluated
y = 4 #y-coordinate evaluated
p = 1.3*10**(-4) #breathing rate (m^3/s)
k = 0.0069 #infectivity constant for dose-response model
P_r = 0.5 #risk tolerance
d = 1.7*10**(-4) #deactivation rate
s = 1.1*10**(-4) #settling rate

Q = np.linspace(0.1,6,60) #ventilation: air exchanges per hour
Q = Q / 3600 #convert to air exchanges per second
V = l * w * h #volume of the room
K = 0.39 * V * Q * (2 * V *0.059)**(-1/3) #eddy diffusion coefficient

#time-axis
t_end = 60*60*time #in seconds
delta_t = 1 #(s) time-steps
n_t = int(t_end/delta_t)
t = np.linspace(delta_t,t_end,n_t)
TTPI = np.zeros_like(Q) #initialise TTPI array

#initialize source function. multiply by delta_t to discretize the function.
S = delta_t * np.full(len(t), R)

#define Impulse function I
m = int(v*3600/(2*l) *time) #no of times the particles travel around the recirculation loop
def I(x,y,x_o,y_o,Q,K):
    #I_y is an approximation of the sum of y-exponentials in the impulse function
    I_y = np.exp(-((y-y_o)**2)/(4*K*t)) + np.exp(-((y+y_o)**2)/(4*K*t))
    for n in range(1,4):
        I_y += np.exp(-((y-y_o - 2*n*w)**2)/(4*K*t)) + np.exp(-((y+y_o + 2*n*w)**2)/(4*K*t))
        I_y += np.exp(-((y-y_o + 2*n*w)**2)/(4*K*t)) + np.exp(-((y+y_o - 2*n*w)**2)/(4*K*t))
    #I_x is an approximation of the sum of x-exponentials in the impulse function
        I_x = np.exp(-((x-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t)**2)/(4*K*t))
    for n in range(1,m+1):
        I_x += np.exp(-((x-x_o -v*t + 2*n*l)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t - 2*n*l)**2)/(4*K*t))
        I_x += np.exp(-((x-x_o -v*t - 2*n*l)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t + 2*n*l)**2)/(4*K*t))
    sinks = np.exp(-(Q+d+s)*t) #effect of ventilation, deactivation and settling sinks
    return 1/(4*np.pi*K*t) * I_y * I_x * sinks #Impulse function

for i in range(len(Q)):
    I_Q = I(x,y,x_o,y_o,Q[i],K[i])
    #Convolution for C
    C = convolve(S,I_Q)[0:len(t)] / (h/2)
    #Calculate dose
    dose = p * cumtrapz(C,t)
    P = (1 - np.exp(-dose*k))

    if P[-1] < 0.5: print("needs longer eval time")
    else: 
        n = 0
        while P[n] < P_r: n += 1
        TTPI[i] = t[n]

TTPI = TTPI/60 #convert to minutes
Q = Q*3600 #convert Q back to ACH

#plot 
plt.plot(Q,TTPI)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Air changes per hour")
plt.ylabel("Time to probable infection")
plt.show()
