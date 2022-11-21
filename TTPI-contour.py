import numpy as np #import numpy as np, to enable numpy arrays and numpy meshgrid
import matplotlib.pyplot as plt #import matplotlib.pyplot as plt to plot graphs
from scipy.signal import convolve #import convolve function from scipy.signal library
from scipy.integrate import cumtrapz #import cumtrapz function from scipy.integration library to preform numerical integration

#Parameters values -- User inputs floats
time_max = 2 # MAX TTPI that can be calculated!
l = 8 #x-length (m)
w = 8 #y-length (m)
h = 3 #z-length (m)
x_o = 4 #x-coordinate of source
y_o = 4 #y-coordinate of source
R = 5 #aerosol generation rate (particles/s)
v = 0.15 #air velocity (m/s) from left to right. 
p = 1.3*10**(-4) #breathing rate (m^3/s)
Q = 2*10**(-4) # Air exchange rate (s^-1)
K = 8.3*10**(-3) # Eddy diffusion coefficient (m^2/s)
d = 1.7*10**(-4) #deactivation rate (s^-1)
s = 1.1*10**(-4) #settling rate (s^-1)
D = 100 #median infectious dose
delta_x = 0.05 #(m) mesh-size
delta_t = 1 #(s) time-steps

#set up mesh
n_x = int(l / delta_x) + 1 #int: calculate number of x-steps
n_y = int(w / delta_x) + 1 #int:calculate number of y-steps
x = np.linspace(0,l,n_x) #define numpy array for x-axis
y = np.linspace(0,w,n_y) #define numpy array for y-axis
X,Y = np.meshgrid(x,y) #define numpy meshgrid for X,Y
T = np.zeros_like(X) #Initialise numpy array of same size as X for T (TTPI)

#time-axis
t_end = 60*60*time_max #float: convert event duration to secondss
n_t = int(t_end/delta_t)  #int: calculate number of time steps
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
m = int(v*3600/(2*l) *time_max) #Calculates m, the number of times the particles travel around the recirculation loop
#For each value in x, calculate the corresponding I_x = Sum [ exp(-(x-x_o+2nl)**2)/(4Kt) + exp(-(x+x_o+2nl)**2)/(4Kt) ]
for i in range(len(x)):
    I_x[i] = np.exp(-((x[i]-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t)**2)/(4*K*t))
    for n in range(1,m+1):
        I_x[i] += np.exp(-((x[i]-x_o -v*t + 2*n*l)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t - 2*n*l)**2)/(4*K*t))
        I_x[i] += np.exp(-((x[i]-x_o -v*t - 2*n*l)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t + 2*n*l)**2)/(4*K*t))
	
sinks = np.exp(-(Q+d+s)*t) #numpy array of size len(t). For each value of t, calculate the effect of ventilation, deactivation and settling sinks = exp(-(Q+d+s)*t)

def is_TTPI_achieved(dose):
    """ this function takes the dose inhaled results for the time specified at
        the start of this program, and then checks if the dose exceeds the mean
        infectious dose within this time.
        If it doesn't, it will raise an exception.
        [added 21/11/2022]

        input: dose (1D np.array of floats)
        return: (int) the index at which the dose results first achieves 
                the mean infectious dose
    """
    if dose[-1] < D:
        raise Exception("The TTPI is not achieved at some positions in the room.")
    else:
        n = 0
        while dose[n] < D:
            n += 1
        return int(n)

# For each mesh point (x,y), calculate the corresponding Impulse function, I [an array of size len(t)]. 
# Then use Scipy Convolve function for S and I to calculate the C at (x,y)
# Use cumtrapz function to calculate dose (array of size len(t)-1) inhaled at (x.y)
# Finally search for t-value when dose > D, ie. when median infectious dose is reached
# if TTPI is not achieved within the room, for plotting purposes, the TTPI is assigned to (time+1) hours.
for i in range(len(x)):
    for j in range(len(y)):
        I = 1/(4*np.pi*K*t) * I_y[j] * I_x[i] * sinks
        C = convolve(S,I)[0:len(t)] / (h/2)
        dose = p * cumtrapz(C,t)
        
        """ [updated 21/11/2022] """
        try:
            n_TTPI = is_TTPI_achieved(dose) + 1
            T[j][i] = t[n_TTPI]
        except:
            # TTPI is not achieved within the room
            # for graph plotting purposes, the TTPI is assigned to (time+1) hours.
            T[j][i] = (time_max+1) * 3600
        """ [updated 21/11/2022] """

        """ Original code when Paper was published.
        n = 0
		while dose[n]<D: n += 1
		T[j][i] = t[n+1]
        """

T = T/60 #convert TTPI to minutes

max_T = max(T[0])
min_T = min(T[0])
for i in range(1, len(x)):
    max_T = max(max_T, max(T[i]))
    min_T = min(min_T, min(T[i]))

if min_T == max_T:
    print("The TTPI everywhere in the room is larger than {} hours.".format(time_max))
else:
    # assign the contour levels desired.    
    # By default, it splits it into 4 levels from minimum TTPI to time_max.
    contour_levels = np.linspace(min_T, time_max*60, 4) 

    #Uses Matplotlib.pyplot to produce a black and white contour plot of X,Y,T
    fig,ax = plt.subplots(1,1)
    cp = ax.contour(X,Y,T,colors='black',levels=contour_levels)
    #Forces the graph produced to be in the shape of a square
    plt.axis('square') #comment this line out if the room is not square
    ax.tick_params(axis='both',labelsize=12) #changes the axis font size to 12
    plt.clabel(cp, levels=contour_levels, fontsize=12, fmt='%1.0f',colors='black')  #Change the contour labels to floats of 0 decimal place with fontsize 12
    plt.show()