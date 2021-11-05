import numpy as np #import numpy as np, to enable numpy arrays and numpy meshgrid
import matplotlib.pyplot as plt #import matplotlib.pyplot as plt to plot graphs
from scipy.signal import convolve #import convolve function from scipy.signal library
from scipy.integrate import cumtrapz #import cumtrapz function from scipy.integration library to preform numerical integration

#Parameters values -- User inputs floats
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
delta_t = 1 #(s) time-steps

#time-axis
t_end = 60*60*time #float: convert event duration to seconds
n_t = int(t_end/delta_t) #int: calculate number of time steps
t = np.linspace(delta_t,t_end,n_t) #define numpy array for time axis

#initialize source function with numpy array of same size as time axis. S is a constant function, so every element is the same level
#S is multiplied by delta_t to discretize the function.
S = delta_t * np.full(len(t), R)

#Define the impulse function
#I_y is a numpy array of size len(t) -- representing an approximation of the sum of y-exponentials in the impulse function.
#For value in t, calculate I_y = Sum [ exp(-(y-y_o+2nw)**2)/(4Kt) + exp(-(y+y_o+2nw)**2)/(4Kt) ]
I_y = np.exp(-((y-y_o)**2)/(4*K*t)) + np.exp(-((y+y_o)**2)/(4*K*t))
for n in range(1,4):
    I_y += np.exp(-((y-y_o - 2*n*w)**2)/(4*K*t)) + np.exp(-((y+y_o + 2*n*w)**2)/(4*K*t))
    I_y += np.exp(-((y-y_o + 2*n*w)**2)/(4*K*t)) + np.exp(-((y+y_o - 2*n*w)**2)/(4*K*t))
#I_x is a numpy array of size len(t) -- representing an approximation of the sum of x-exponentials in the impulse function.
#For each value of t, calculate the corresponding I_x = Sum [ exp(-(x-x_o+2nl)**2)/(4Kt) + exp(-(x+x_o+2nl)**2)/(4Kt) ]I_x = np.exp(-((x-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t)**2)/(4*K*t))
m = int(v*3600/(2*l) *time) #Calculates m, the number of times the particles travel around the recirculation loop
for n in range(1,m+1):
    I_x += np.exp(-((x-x_o -v*t + 2*n*l)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t - 2*n*l)**2)/(4*K*t))
    I_x += np.exp(-((x-x_o -v*t - 2*n*l)**2)/(4*K*t)) + np.exp(-((x+x_o+v*t + 2*n*l)**2)/(4*K*t))

sinks = np.exp(-(Q+d+s)*t)  #numpy array of size len(t). Calculates the effect of ventilation, deactivation and settling sinks

I = 1/(4*np.pi*K*t) * I_y * I_x * sinks #Impulse function at (x,y) - numpy array of size len(t)

#use Scipy Convolve function for S and I to calculate the C at (x,y)
C = convolve(S,I)[0:len(t)] / (h/2)
#use Scipy cumtrapz function to calculate the dose (array of len(t)-1) at (x,y)
dose = p * cumtrapz(C,t)
#calculate P (array of len(t)-1)
P = 1 - np.exp(-dose * k)

t = t/60 #convert time-axis to minutes for plotting of graph

P = P * 100 #convert probability to percent for plotting of graph
#Uses Matplotlib.pyplot to produce a line graph P vs t[1:]
plt.plot(t[1:],P)
#Change axis fontsize to 12
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#label axes
plt.xlabel("Time (min)")
plt.ylabel("Probability ($\%$)")
plt.show()
