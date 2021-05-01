import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.integrate import cumtrapz

#parameters
time = 3 #event duration, in hours
l = 8 #8 #x-length (m)
w = 8 #8 #y-length (m)
h = 3 #z-length (m)
x_o = 4 #x-coordinate of source
y_o = 4	 #y-coordinate of source
R = 5 #aerosol generation rate (particles/s)
v = 0.15 #air velocity (m/s) from left to right. 
p = 1.3*10**(-4) #breathing rate (m^3/s)
k = 0.0069 #infectivity constant for dose-response model
Q = 0.0017 # Air exchange rate (s^-1)
K = 0.044 # Eddy diffusion coefficient (m^2/s)
d = 1.7*10**(-4) #deactivation rate (s^-1)
s = 1.1*10**(-4) #settling rate (s^-1)

#set up mesh
delta_x = 0.1 #(m) mesh-size
n_x = int(l / delta_x) + 1
n_y = int(w / delta_x) + 1
x = np.linspace(0,l,n_x)
y = np.linspace(0,w,n_y) 
X,Y = np.meshgrid(x,y)

#time-axis
t_end = 60*60*time #in seconds
delta_t = 1 #(s) time-steps
n_t = int(t_end/delta_t)
t = np.linspace(delta_t,t_end,n_t)

#initialise C and P arrays
C = np.empty([len(x)*len(y),len(t)])
P = np.empty([len(x)*len(y),len(t)-1])
avg_P = np.zeros(len(t)-1)
avg_C = np.zeros(len(t))

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
m = int(v*3600/(2*l) *time) #no of times the particles travel around the recirculation loop
for i in range(len(x)):
    I_x[i] = np.exp(-((x[i]-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t)**2)/(4*K*t))
    for n in range(1,m+1):
        I_x[i] += np.exp(-((x[i]-x_o -v*t + 2*n*l)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t - 2*n*l)**2)/(4*K*t))
        I_x[i] += np.exp(-((x[i]-x_o -v*t - 2*n*l)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t + 2*n*l)**2)/(4*K*t))
sinks = np.exp(-(Q+d+s)*t) #effect of ventilation, deactivation and settling sinks

n=0
for i in range(len(x)):
    for j in range(len(y)):
        #Calculate impulse for each mesh point
        I = 1/(4*np.pi*K*t) * I_y[j] * I_x[i] * sinks
        #Convolve S and I to find C
        C[n] = convolve(S,I)[0:len(t)] / (h/2)
        #calculate the dose inhaled
        dose = p * cumtrapz(C[n],t)
        #calculate the probability
        P[n] = 1 - np.exp(-dose*k)
        n += 1

for i in range(len(t)-1):
	sum_P = 0
	sum_C = 0
	for n in range(len(x)*len(y)): 
		sum_P += P[n][i]
		sum_C += C[n][i+1]
	avg_P[i] = sum_P/(len(x)*len(y))
	avg_C[i+1] = sum_C/(len(x)*len(y))

P_avgC = 1 - np.exp(-p*cumtrapz(avg_C,t)*k)

I_max = 1/(4*np.pi*K*t) *sinks* I_y[int(w/delta_x/2)] * I_x[int(l/delta_x/2 + 1/delta_x)]
I_min = 1/(4*np.pi*K*t) * I_y[-1] * I_x[-1] * sinks
C_max = convolve(S,I_max)[0:len(t)]  / (h/2)
C_min = convolve(S,I_min)[0:len(t)]  / (h/2)
dose_max = p*cumtrapz(C_max,t)
dose_min = p*cumtrapz(C_min,t)
P_max = 1 - np.exp(-dose_max*k)
P_min = 1 - np.exp(-dose_min*k)

t_plot = t[1:]/60
plt.plot(t_plot,avg_P,label='average Probability',color='red',linestyle='solid',linewidth=2)
plt.plot(t_plot,P_max,label='max Probability',color='red',linestyle='dotted',linewidth=2)
plt.plot(t_plot,P_min,label='min Probability',color='red',linestyle='dotted',linewidth=2)
plt.plot(t_plot,P_avgC,label='Prob-averaged C',color='blue',linestyle='dashed',linewidth=2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
