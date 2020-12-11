import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.integrate import cumtrapz

v = 0.15
N_th = 150000
R = np.arange(0.1,10.1,0.1)

# infinite room case
K = 0.18
T_inf = np.zeros_like(R)

#time steps
t_end = 60
delta_t = 0.3
n_t = int(t_end/delta_t) + 1
t = np.linspace(0.1,t_end+0.1,n_t)

I = 1/(4*np.pi*K*t) * np.exp(-((1-v*t)**2)/(4*K*t))
for i in range(len(R)):
    S = np.full(len(t), R[i])
    C = convolve(S,I,mode="valid") * delta_t
    TTI = N_th / C
    T_inf[i] = TTI / 60  

#finite rooms
Q_values = [0,0.0002,0.00083,0.0017]
Q = Q_values[1]
eval = 20 #most difficult to find the right time here for small values of R.
TTI_limit = 180
# setting up room dimensions: 4x4 (personal room), 8x8 (classroom), 30x15 (basketball court/auditorium); 105x68 (football field)
h = 3
l = np.array([4,8,30,105])
b = np.array([4,8,15,68])
V = l * b * h
K = V * Q * (2*V)**(-1/3)
T = []
for i in range(len(l)):
    T.append(np.zeros_like(R)) 

#source: centre; evaluated at 1m downwind.
x_o = l/2
y_o = b/2
x = x_o
y = y_o + 1

#set up time
t_end = 60*60 * eval
delta_t = 0.3
n_t = int(t_end/delta_t) + 1
t = np.linspace(0.1,t_end+0.1,n_t)

for i in range(len(l)):
    k = int( v * 3600 / (2*l[i]) * eval)  #nodes
    C_y = np.exp(-((y[i]-y_o[i])**2)/(4*K[i]*t)) + np.exp(-((y[i]+y_o[i])**2)/(4*K[i]*t))
    C_x = np.exp(-((x[i]-x_o[i]-v*t)**2)/(4*K[i]*t)) + np.exp(-((x[i]+x_o[i]+v*t)**2)/(4*K[i]*t))
    for n in range(1,k+1):
        C_y = C_y + np.exp(-((y[i]-y_o[i] - 2*n*l[i])**2)/(4*K[i]*t)) + np.exp(-((y[i]+y_o[i] + 2*n*l[i])**2)/(4*K[i]*t))
        C_y = C_y + np.exp(-((y[i]-y_o[i] + 2*n*l[i])**2)/(4*K[i]*t)) + np.exp(-((y[i]+y_o[i] - 2*n*l[i])**2)/(4*K[i]*t))
        C_x = C_x + np.exp(-((x[i]-x_o[i] -v*t - 2*n*b[i])**2)/(4*K[i]*t)) + np.exp(-((x[i]+x_o[i]+v*t + 2*n*b[i])**2)/(4*K[i]*t))
        C_x = C_x + np.exp(-((x[i]-x_o[i] -v*t + 2*n*b[i])**2)/(4*K[i]*t)) + np.exp(-((x[i]+x_o[i]+v*t - 2*n*b[i])**2)/(4*K[i]*t))
    I = 1/(4*np.pi*K[i]*t) * C_y * C_x * np.exp(-Q*t)
    
    for j in range(len(R)):
        S = np.full(len(t), R[j])      
        C = convolve(S,I)[0:len(t)] * delta_t
        
        C_int = cumtrapz(C,t,initial=0) #array
        n = len(t) - 1
        while C_int[n] > N_th: n = n-1
        T[i][j] = t[n] / 60

txt_name = "./data/TTI-vs-R.txt"
myfile = open(txt_name,"w")
myfile = open(txt_name,"a")
for i in range(len(l)):
    j = 0
    while j < len(R)-1:
        words = str(T[i][j]) + ","
        myfile.write(words)
        j = j + 1
    myfile.write(str(T[i][j]) + "\n")
myfile.close()

#plot 
plt.loglog(R,T_inf,linestyle='dotted',color='black')
plt.loglog(R,T[0],linestyle='dashed',color='black')
plt.loglog(R,T[1],linestyle='solid',color='black')
plt.loglog(R,T[2],linestyle='dashdot',color='black')
plt.loglog(R,T[3],linestyle='dashed',color='black')
#plt.ylim([0,240])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plot_name = "./plots/TTI-vs-R.png"
plt.savefig(plot_name)
plt.show()
