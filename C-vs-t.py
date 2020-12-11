import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

#PDE parameters
v = 0.15 
R = 1
Q = [0.000033, 0.0002, 0.00083, 0.0017]
K = [0.00088, 0.0053, 0.022, 0.044] 

#Domain
l = 8 
b = 8  

#source's position (currently the centre)
x_o = 4
y_o = 4

x = 5
y = 4
hour = 8

# graph's axis
t_end = 60*60 * hour
delta_t = 0.3
n_t = int(t_end/delta_t) + 1
t = np.linspace(0.1,t_end+0.1,n_t)

C = [np.zeros_like(t),np.zeros_like(t),np.zeros_like(t),np.zeros_like(t)]

#set up convolution
S = np.full(len(t), R)

k = int(34*hour) #nodes 34 for 1h

for i in range(len(Q)):
    C_y = np.exp(-((y-y_o)**2)/(4*K[i]*t)) + np.exp(-((y+y_o)**2)/(4*K[i]*t))
    C_x = np.exp(-((x-x_o-v*t)**2)/(4*K[i]*t)) + np.exp(-((x+x_o+v*t)**2)/(4*K[i]*t))
    for n in range(1,k+1):
        C_y = C_y + np.exp(-((y-y_o - 2*n*l)**2)/(4*K[i]*t)) + np.exp(-((y+y_o + 2*n*l)**2)/(4*K[i]*t))
        C_y = C_y + np.exp(-((y-y_o + 2*n*l)**2)/(4*K[i]*t)) + np.exp(-((y+y_o - 2*n*l)**2)/(4*K[i]*t))
        C_x = C_x + np.exp(-((x-x_o -v*t - 2*n*b)**2)/(4*K[i]*t)) + np.exp(-((x+x_o+v*t + 2*n*b)**2)/(4*K[i]*t))
        C_x = C_x + np.exp(-((x-x_o -v*t + 2*n*b)**2)/(4*K[i]*t)) + np.exp(-((x+x_o+v*t - 2*n*b)**2)/(4*K[i]*t))
    I = 1/(4*np.pi*K[i]*t) * C_y * C_x * np.exp(-Q[i]*t)
    C[i] = convolve(S,I)[0:len(t)] * delta_t * 0.5

#n_s = int(60/delta_t)
t = t / 60

#plot 
plt.plot(t,C[0],linestyle='dashed',color='red')
plt.plot(t,C[1],linestyle='solid',color='orange')
plt.plot(t,C[2],linestyle='dashdot',color='green')
plt.plot(t,C[3],linestyle='dotted',color='blue')
#plt.loglog(t[n_s:],C[0][n_s:],linestyle='dashed',color='red')
#plt.loglog(t[n_s:],C[1][n_s:],linestyle='solid',color='orange')
#plt.loglog(t[n_s:],C[2][n_s:],linestyle='dashdot',color='green')
#plt.loglog(t[n_s:],C[3][n_s:],linestyle='dotted',color='blue') 
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plot_name = "./plots/C-vs-t-XY" +str(x) + str(y) + ".png"
plt.savefig(plot_name)
plt.show()
