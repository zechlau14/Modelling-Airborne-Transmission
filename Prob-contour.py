import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.integrate import trapz

#parameters
time = 1 #event duration, in hours
l = 8 #x-length (m)
w = 8 #y-length (m)
h = 3 #z-length (m)
x_o = 4 #x-coordinate of source
y_o = 4 #y-coordinate of source
R = 5 #aerosol generation rate (particles/s)
v = 0.15 #air velocity (m/s) from left to right. 
p = 1.3*10**(-4) #breathing rate (m^3/s)
k = 0.0069 #infectivity constant for dose-response model
Q = 0.0002 # Air exchange rate (s^-1)
K = 0.0053 # Eddy diffusion coefficient (m^2/s)
d = 1.7*10**(-4) #deactivation rate (s^-1)
s = 1.1*10**(-4) #settling rate (s^-1)

#set up mesh
delta_x = 0.05 #(m) mesh-size
n_x = int(l / delta_x) + 1
n_y = int(w / delta_x) + 1
x = np.linspace(0,l,n_x)
y = np.linspace(0,w,n_y) 
X,Y = np.meshgrid(x,y)
P = np.zeros_like(X) #Initialise P (probability)

#time-axis
t_end = 60*60*time #in seconds
delta_t = 1 #(s) time-steps
n_t = int(t_end/delta_t)
t = np.linspace(delta_t,t_end,n_t)

#initialize source function. multiply by delta_t to discretize the function.
S = delta_t * np.full(len(t), R)

#impulse function:
#I_y is an approximation of the sum of y-exponentials in the impulse function
I_y = [np.zeros_like(t)]
for j in range(1,len(y)): I_y.append(np.zeros_like(t))
for j in range(len(y)):
    I_y[j] = np.exp(-((y[j]-y_o)**2)/(4*K*t)) + np.exp(-((y[j]+y_o)**2)/(4*K*t))
    for n in range(1,4):
        I_y[j] += np.exp(-((y[j]-y_o - 2*n*w)**2)/(4*K*t)) + np.exp(-((y[j]+y_o + 2*n*w)**2)/(4*K*t))
        I_y[j] += np.exp(-((y[j]-y_o + 2*n*w)**2)/(4*K*t)) + np.exp(-((y[j]+y_o - 2*n*w)**2)/(4*K*t))
#I_x is an approximation of the sum of x-exponentials in the impulse function
I_x = [np.zeros_like(t)]
for i in range(1,len(x)): I_x.append(np.zeros_like(t))
m = int(v*3600/(2*l) *time)  #no of times the particles travel around the
for i in range(len(x)):
    I_x[i] = np.exp(-((x[i]-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t)**2)/(4*K*t))
    for n in range(1,m+1):
        I_x[i] += np.exp(-((x[i]-x_o -v*t + 2*n*l)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t - 2*n*l)**2)/(4*K*t))
        I_x[i] += np.exp(-((x[i]-x_o -v*t - 2*n*l)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t + 2*n*l)**2)/(4*K*t))
sinks = np.exp(-(Q+d+s)*t) #effect of ventilation, deactivation and settling sinks

for i in range(len(x)):
    for j in range(len(y)):
    #for j in range(int(len(y)/2+1)):
        #Calculate impulse for each mesh point
        I = 1/(4*np.pi*K*t) * I_y[j] * I_x[i] * sinks
        #Convolve S and I
        C = convolve(S,I)[0:len(t)] / (h/2)
        #calculate the dose inhaled
        dose = p * trapz(C,t)
        #calculate the probability
        P[j][i] = 1 - np.exp(-dose*k)

#plot probability contours
fig,ax = plt.subplots(1,1)
cp = ax.contour(X,Y,P,colors='black')
plt.axis('square') #comment this line out if the room is not square
ax.tick_params(axis='both',labelsize=12)
plt.clabel(cp,fontsize=12, levels=7, fmt='%1.3f',colors='black')
plt.show()
