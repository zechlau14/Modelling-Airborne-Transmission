import numpy as np #import numpy as np, to enable numpy arrays and numpy meshgrid
import matplotlib.pyplot as plt #import matplotlib.pyplot as plt to plot graphs
from scipy.signal import convolve #import convolve function from scipy.signal library
from scipy.integrate import cumtrapz #import cumtrapz function from scipy.integration library to preform numerical integration

#Parameters values -- User inputs floats
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
delta_t = 1 #(s) time-steps

#Initiate Q array, and the corresponding K values
Q = np.linspace(0.1,6,60) #ventilation: air exchanges per hour
Q = Q / 3600 #convert to air exchanges per second
V = l * w * h #volume of the room
K = 0.39 * V * Q * (2 * V *0.059)**(-1/3) #Calculate eddy diffusion coefficient (m/s) for each Q

#time-axis
t_end = 60*60*time #float: convert event duration to seconds
n_t = int(t_end/delta_t) #int: calculate number of time steps
t = np.linspace(delta_t,t_end,n_t) #define numpy array for time axis
TTPI = np.zeros_like(Q) #initialise TTPI array of size len(Q)

#initialize source function with numpy array of same size as time axis. S is a constant function, so every element is the same level
#S is multiplied by delta_t to discretize the function.
S = delta_t * np.full(len(t), R)

m = int(v*3600/(2*l) *time) #Calculates m, the number of times the particles travel around the recirculation loop
#define Impulse function I
def I(x,y,x_o,y_o,Q,K):
    #I_y is a numpy array of size len(t) -- representing an approximation of the sum of y-exponentials in the impulse function.
    #For value in t, calculate I_y = Sum [ exp(-(y-y_o+2nw)**2)/(4Kt) + exp(-(y+y_o+2nw)**2)/(4Kt) ]
    I_y = np.exp(-((y-y_o)**2)/(4*K*t)) + np.exp(-((y+y_o)**2)/(4*K*t))
    for n in range(1,4):
        I_y += np.exp(-((y-y_o - 2*n*w)**2)/(4*K*t)) + np.exp(-((y+y_o + 2*n*w)**2)/(4*K*t))
        I_y += np.exp(-((y-y_o + 2*n*w)**2)/(4*K*t)) + np.exp(-((y+y_o - 2*n*w)**2)/(4*K*t))
    #I_x is a numpy array of size len(t) -- representing an approximation of the sum of x-exponentials in the impulse function.
    #For each value of t, calculate the corresponding I_x = Sum [ exp(-(x-x_o+2nl)**2)/(4Kt) + exp(-(x+x_o+2nl)**2)/(4Kt) ]I_x = np.exp(-((x-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t)**2)/(4*K*t))
    I_x = np.exp(-((x-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t)**2)/(4*K*t))
    for n in range(1,m+1):
        I_x += np.exp(-((x-x_o -v*t + 2*n*l)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t - 2*n*l)**2)/(4*K*t))
        I_x += np.exp(-((x-x_o -v*t - 2*n*l)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t + 2*n*l)**2)/(4*K*t))
    sinks = np.exp(-(Q+d+s)*t)  #numpy array of size len(t). Calculates the effect of ventilation, deactivation and settling sinks
    return 1/(4*np.pi*K*t) * I_y * I_x * sinks #Impulse function

#For each value of Q, calculate I.
#use Scipy Convolve function for S and I to calculate the C for each Q
#use Scipy cumtrapz function to calculate the dose (array of len(t)-1) for each Q
#calculate P (array of len(t)-1) for each Q
for i in range(len(Q)):
    I_Q = I(x,y,x_o,y_o,Q[i],K[i])
    C = convolve(S,I_Q)[0:len(t)] / (h/2)
    dose = p * cumtrapz(C,t)
    P = (1 - np.exp(-dose*k))
    #search Prob to find time-step when Prob is first > 0.5, ie. the TTPI
    if P[-1] < 0.5: print("needs longer eval time")
    else: 
        n = 0
        while P[n] < P_r: n += 1
        TTPI[i] = t[n]

TTPI = TTPI/60 #convert TTPI to minutes
Q = Q*3600 #convert Q back to ACH

#Uses Matplotlib.pyplot to produce a plot of TTPI vs Q 
plt.plot(Q,TTPI)
#Change axis fontsize to 12
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#label axes
plt.xlabel("Air changes per hour")
plt.ylabel("Time to probable infection")
plt.show()
