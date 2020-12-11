import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.integrate import cumtrapz

#PDE parameters
V = 192
v = 0.15
R = [5,2.5]
#R = [0.5,0.25]
N_th = 150000
Q = np.arange(0.01,6.01,0.01)
#Q = np.arange(0.01,3.01,0.01)
Q = Q / 3600
K = V * Q * (2*V)**(-1/3)
T = [np.zeros_like(Q),np.zeros_like(Q)]

#Domain
l = 8 
b = 8  

#source's position (currently the centre)
x_o = 4
y_o = 4

x = 8
y = 8
eval = 6 #Evaluation time to be found using TTI-point
#10.5 #7.5 #6 #1.5

# graph's axis
t_end = 60*60 * eval
delta_t = 0.3
n_t = int(t_end/delta_t) + 1
t = np.linspace(0.1,t_end+0.1,n_t)

for j in range(len(R)):

    for i in range(len(Q)):
        S = np.full(len(t), R[j])
        C_y = np.exp(-((y-y_o)**2)/(4*K[i]*t)) + np.exp(-((y+y_o)**2)/(4*K[i]*t))
        C_x = np.exp(-((x-x_o-v*t)**2)/(4*K[i]*t)) + np.exp(-((x+x_o+v*t)**2)/(4*K[i]*t))
        k = int(34*eval) #nodes 34 for 1h
        for n in range(1,k+1):
            C_y = C_y + np.exp(-((y-y_o - 2*n*l)**2)/(4*K[i]*t)) + np.exp(-((y+y_o + 2*n*l)**2)/(4*K[i]*t))
            C_y = C_y + np.exp(-((y-y_o + 2*n*l)**2)/(4*K[i]*t)) + np.exp(-((y+y_o - 2*n*l)**2)/(4*K[i]*t))
            C_x = C_x + np.exp(-((x-x_o -v*t - 2*n*b)**2)/(4*K[i]*t)) + np.exp(-((x+x_o+v*t + 2*n*b)**2)/(4*K[i]*t))
            C_x = C_x + np.exp(-((x-x_o -v*t + 2*n*b)**2)/(4*K[i]*t)) + np.exp(-((x+x_o+v*t - 2*n*b)**2)/(4*K[i]*t))
        I = 1/(4*np.pi*K[i]*t) * C_y * C_x * np.exp(-Q[i]*t)
        C = convolve(S,I)[0:len(t)] * delta_t
        C_int = cumtrapz(C,t,initial=0) #array

        n = len(t) - 1
        while C_int[n] > N_th: n = n-1
        T[j][i] = t[n] / (60)

file_name = "TTI-vs-Q-XY" + str(x) + str(y) + "talk"
txt_name = "./data/" + file_name + ".txt"
myfile = open(txt_name,"w")
myfile = open(txt_name,"a")
for j in range(len(R)):
    for i in range(len(Q)):
        words = str(T[j][i]) + ","
        myfile.write(words)
myfile.close()

"""q_case = [0.12,0.72,3,6]
i = 0
T_case = [T[i][11], T[i][71], T[i][299]]#, T[i][599]]
T_case_mask = [ T[i+1][11], T[i+1][71], T[i+1][299]]#, T[i+1][599]] """

#plot 
plt.plot(3600*Q,T[0],linestyle='solid',color='red')
plt.plot(3600*Q,T[1],linestyle='dashed',color='blue')
#plt.scatter(q_case,T_case,color='red',marker="x")
#plt.scatter(q_case,T_case_mask,color='blue',marker="x")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plot_name = "./plots/" + file_name + ".png"
plt.savefig(plot_name)
plt.show()
