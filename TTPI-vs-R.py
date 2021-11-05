import numpy as np #import numpy as np, to enable numpy arrays and numpy meshgrid
import matplotlib.pyplot as plt #import matplotlib.pyplot as plt to plot graphs
from scipy.signal import convolve #import convolve function from scipy.signal library
from scipy.integrate import trapz #import trapz function from scipy.integrate for numerical integration

#Parameters values -- User inputs floats
v = 0.15 #air velocity (m/s)
Q = [0.000033, 0.0002, 0.00083, 0.0017] #the air changes per second (s^-1)
K = [0.00088, 0.0053, 0.022, 0.044] #eddy diffusion coefficients (m^2/s)
d = 1.7*10**(-4) #deactivation rate (s^-1)
s = 1.1*10**(-4) #settling rate (s^-1)
R = np.linspace(1,7.5,66)
time = 24  #in hours
w = 8 #y-length (m)
l = 8 #x-length (m)
h = 3 #z-length (m)
x_o = 4 #x-coordinate of source
y_o = 4 #y-coordinate of source
x = 5 #x-coordinate of point evaluated at
y = 4 #y-coordinate of point evaluated at
p = 1.3*10**(-4) #breathing rate (s^-1)
k = 0.0069 #constant for dose-response model
P_r = 0.5 #Target probability
delta_t = 1 #(s) time-steps

#time-axis
t_end = 60*60*time #float: convert event duration to seconds
n_t = int(t_end/delta_t) #int: calculate number of time steps
t = np.linspace(delta_t,t_end,n_t) #define numpy array for time axis
TTPI = [np.zeros_like(R),np.zeros_like(R),np.zeros_like(R),np.zeros_like(R)] #initialise TTPI array of size 4 x len(R)

#initialize source function with numpy array of same size as time axis. S is a constant function, so every element is the same level
#S is multiplied by delta_t to discretize the function.
S = np.full(len(t), 1) *delta_t

m = int(v*3600/(2*l) *time)  #Calculates m, the number of times the particles travel around the recirculation loop

#For each Q value, define the impulse function, I (array of len(t))
#For each value in t, calculate the corresponding C_x = Sum [ exp(-(x-x_o+2nl)**2)/(4Kt) + exp(-(x+x_o+2nl)**2)/(4Kt) ]
#For each value in t, calculate the corresponding C_y = Sum [ exp(-(y-y_o+2nw)**2)/(4Kt) + exp(-(y+y_o+2nw)**2)/(4Kt) ]
#then combine them into the impulse function I = 1/(4*np.pi*K*t) * C_y * C_x * np.exp(-(Q+d+s)*t)
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
    #use Scipy Convolve function for S and I to calculate the C at (x,y)
    C = convolve(S,I)[0:len(t)] / (h/2)
    #Use cumtrapz function to calculate dose inhaled at (x.y)
    dose = p * cumtrapz(C,t)

    for j in range(len(R)):
        #For each R value, calculate the Probability (array of len(R) x (len(t)-1) )
        Prob = (1 - np.exp(-R[j]*dose*k))
        #search each Prob to find time-step when Prob is first > P_r
        if Prob[-1] < P_r: print("needs longer eval time")
        else: 
            n = 0
            while Prob[n] < P_r: n += 1
            TTPI[i][j] = t[n]/60

#Uses Matplotlib.pyplot to produce a plot of TTPI vs R
plt.loglog(R,TTPI[0],linestyle='dashed',color='red',linewidth=2)
plt.loglog(R,TTPI[1],linestyle='solid',color='orange',linewidth=2)
plt.loglog(R,TTPI[2],linestyle='dashdot',color='green',linewidth=2)
plt.loglog(R,TTPI[3],linestyle='dotted',color='blue',linewidth=2)
#Change axis fontsize to 12
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#label axes
#plt.xlabel("log$_{10}$ $R$")
#plt.ylabel("TTPI (min)")
plt.show()
