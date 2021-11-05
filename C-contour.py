import numpy as np #import numpy as np, to enable numpy arrays and numpy meshgrid
import matplotlib.pyplot as plt #import matplotlib.pyplot as plt to plot graphs
from scipy.signal import convolve #import convolve function from scipy.signal library

#Parameters values -- User inputs floats
time = 1 #event duration, in hours
l = 8 #x-length (m) of room
w = 8 #y-length (m) of room
h = 3 #z-length (m) of room
x_o = 4 #x-coordinate of source
y_o = 4 #y-coordinate of source
v = 0.15 #air velocity (m/s) from left to right. 
R = 0.5 #aerosol emission rate (particles/s)
Q = 0.0002 # Air exchange rate (s^-1)
K = 0.0053 # Eddy diffusion coefficient (m^2/s)
d = 1.7*10**(-4) #deactivation rate (s^-1)
s = 1.1*10**(-4) #settling rate (s^-1)
delta_x = 0.05 #(m) mesh-size
delta_t = 1 #(s) time-steps

#set up mesh
n_x = int(l / delta_x) + 1 #calculate number of x-steps
n_y = int(w / delta_x) + 1 #calculate number of y-steps
x = np.linspace(0,l,n_x) #define numpy array for x-axis
y = np.linspace(0,w,n_y) #define numpy array for y-axis
X,Y = np.meshgrid(x,y) #define numpy meshgrid for X,Y
C = np.zeros_like(X) #Initialise numpy array of same size as X for C (the concentration)

#time-axis
t_end = 60*60*time #convert event duration to seconds
n_t = int(t_end/delta_t) #calculate number of time steps
t = np.linspace(delta_t,t_end,n_t) #define numpy array for time axis


#initialize source function with numpy array of same size as time axis. S is a constant function, so every element is the same level
#S is multiplied by delta_t to discretize the function.
S = delta_t * np.full(len(t), R)

#Define the impulse function
#I_y is an approximation of the sum of y-exponentials in the impulse function.
#Initialise I_y as an array of size len(y) x len(t)
I_y = [np.zeros_like(t)]
for j in range(1,len(y)): I_y.append(np.zeros_like(t))
#For each value in y, calculate the corresponding I_y = Sum [ exp(-(y-y_o+2nw)**2)/(4Kt) + exp(-(y+y_o+2nw)**2)/(4Kt) ]
for j in range(len(y)):
    I_y[j] = np.exp(-((y[j]-y_o)**2)/(4*K*t)) + np.exp(-((y[j]+y_o)**2)/(4*K*t))
    for n in range(1,4):
        I_y[j] += np.exp(-((y[j]-y_o - 2*n*w)**2)/(4*K*t)) + np.exp(-((y[j]+y_o + 2*n*w)**2)/(4*K*t))
        I_y[j] += np.exp(-((y[j]-y_o + 2*n*w)**2)/(4*K*t)) + np.exp(-((y[j]+y_o - 2*n*w)**2)/(4*K*t))
        
#I_x is an approximation of the sum of x-exponentials in the impulse function
#Initiate I_x as an array of size len(x) x len(t)
I_x = [np.zeros_like(t)]
for i in range(1,len(x)): I_x.append(np.zeros_like(t))
m = int(v*3600/(2*l) *time) #Calculates m, the number of times the particles travel around the recirculation loop
#For each value in x, calculate the corresponding I_x = Sum [ exp(-(x-x_o+2nl)**2)/(4Kt) + exp(-(x+x_o+2nl)**2)/(4Kt) ]
for i in range(len(x)):
    I_x[i] = np.exp(-((x[i]-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t)**2)/(4*K*t))
    for n in range(1,m+1):
        I_x[i] += np.exp(-((x[i]-x_o -v*t + 2*n*l)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t - 2*n*l)**2)/(4*K*t))
        I_x[i] += np.exp(-((x[i]-x_o -v*t - 2*n*l)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t + 2*n*l)**2)/(4*K*t))
        
sinks = np.exp(-(Q+d+s)*t) #numpy array of size len(t). Calculates the effect of ventilation, deactivation and settling sinks

#For each mesh point (x,y), calculate the corresponding Impulse function, I [an array of size len(t)]. 
#Then use Scipy Convolve function for S and I to calculate the C at (x,y)
for i in range(len(x)):
    for j in range(len(y)):      
        I = 1/(4*np.pi*K*t) * I_y[j] * I_x[i] * sinks
        C[j][i] = convolve(S,I,mode='valid') / (h/2) 

#Uses Matplotlib.pyplot to produce a black and white contour plot of X,Y,C
fig,ax = plt.subplots(1,1)
cp = ax.contour(X,Y,C,colors='black',levels=7) 
#Forces the graph produced to be in the shape of a square
plt.axis('square') #comment this line out if the room is not square
ax.tick_params(axis='both',labelsize=12) #changes the axis font size to 12
plt.clabel(cp,fontsize=12, fmt='%1.1f',colors='black') #Change the contour labels to floats of 1 decimal place with fontsize 12
#uncomment the following line for the inclusion of the curly R
#plt.clabel(cp,fontsize=12, fmt='%1.1f $\mathcal{R}$',colors='black')
plt.show() #shows graph
