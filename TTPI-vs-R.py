import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.integrate import cumtrapz

#parameters
v = 0.15 #air velocity (m/s)
Q = [0.000033, 0.0002, 0.00083, 0.0017] #the air changes per second
K = [0.00088, 0.0053, 0.022, 0.044] #eddy diffusion coefficients
d = 1.7*10**(-4) #deactivation rate
s = 1.1*10**(-4) #settling rate
R = np.linspace(1,7.5,66)

time = 24 #48 #24 #in hours
w = 8 #y-length
l = 8 #x-length
h = 3 #z-length
x_o = 4 #x-coordinate of source
y_o = 4 #y-coordinate of source
x = 5 #x-coordinate of point evaluated at
y = 4 #y-coordinate of point evaluated at
p = 1.3*10**(-4) #breathing rate
k = 0.0069 #constant for dose-response model
P_r = 0.5

#time-axis
t_end = 60*60*time #in seconds
delta_t = 1 #(s) time-steps
n_t = int(t_end/delta_t)
t = np.linspace(delta_t,t_end,n_t)
TTPI = [np.zeros_like(R),np.zeros_like(R),np.zeros_like(R),np.zeros_like(R)] #initialise TTPI array

#initialize functions
S = np.full(len(t), 1) *delta_t #source function

m = int(v*3600/(2*l) *time) #no of times the particles travel around the recirculation loop

#Impulse function, I
for i in range(len(Q)):
    C_y = np.exp(-((y-y_o)**2)/(4*K[i]*t)) + np.exp(-((y+y_o)**2)/(4*K[i]*t))
    C_x = np.exp(-((x-x_o-v*t)**2)/(4*K[i]*t)) + np.exp(-((x+x_o+v*t)**2)/(4*K[i]*t))
    for n in range(1,4):
        C_y = C_y + np.exp(-((y-y_o - 2*n*w)**2)/(4*K[i]*t)) + np.exp(-((y+y_o + 2*n*w)**2)/(4*K[i]*t))
        C_y = C_y + np.exp(-((y-y_o + 2*n*w)**2)/(4*K[i]*t)) + np.exp(-((y+y_o - 2*n*w)**2)/(4*K[i]*t))
    for n in range(1,m+1):
        C_x = C_x + np.exp(-((x-x_o -v*t + 2*n*l)**2)/(4*K[i]*t)) + np.exp(-((x+x_o+v*t - 2*n*l)**2)/(4*K[i]*t))
        C_x = C_x + np.exp(-((x-x_o -v*t - 2*n*l)**2)/(4*K[i]*t)) + np.exp(-((x+x_o+v*t + 2*n*l)**2)/(4*K[i]*t))
    I = 1/(4*np.pi*K[i]*t) * C_y * C_x * np.exp(-(Q[i]+d+s)*t)
    #Convolution for C
    C = convolve(S,I)[0:len(t)] / (h/2)
    dose = p * cumtrapz(C,t)

    for j in range(len(R)):
        Prob = (1 - np.exp(-R[j]*dose*k))
        if Prob[-1] < P_r: print("needs longer eval time")
        else: 
            n = 0
            while Prob[n] < P_r: n += 1
            TTPI[i][j] = t[n]/60

#plot 
plt.loglog(R,TTPI[0],linestyle='dashed',color='red',linewidth=2)
plt.loglog(R,TTPI[1],linestyle='solid',color='orange',linewidth=2)
plt.loglog(R,TTPI[2],linestyle='dashdot',color='green',linewidth=2)
plt.loglog(R,TTPI[3],linestyle='dotted',color='blue',linewidth=2)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.xlabel("log$_{10}$ $R$")
#plt.ylabel("TTPI (min)")
plt.show()
