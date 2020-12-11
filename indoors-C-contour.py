import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

#PDE parameters
v = 0.15
R = 1
Q_value = [0.000033, 0.0002, 0.000833, 0.00167]
K_value = [0.00088, 0.0053, 0.0220, 0.0441] 
#[0.000879, 0.00528, 0.0220, 0.0441] 

scenario = 0
Q = Q_value[scenario]
K = K_value[scenario]

hour = 1

#mesh
l= 8
b= 8
delta_x = 0.05 
n_x = int(l / delta_x) + 1
x = np.linspace(0,l,n_x) 
y = np.linspace(0,b,n_x) 
X,Y = np.meshgrid(x,y)
C = np.zeros_like(X)

#source's position
x_o = 4 
y_o = 4

t_end = 60*60*hour 
delta_t = 0.3 # 0.25
n_t = int(t_end/delta_t) + 1
t = np.linspace(0.1,t_end+0.1,n_t)

#set up convolution
#I = np.zeros(len(t))
S = np.full(len(t), R)

m = int(34*hour) #nodes

for i in range(len(x)):
    for j in range(int((len(y)+1)/2)):
        C_y = np.exp(-((y[j]-y_o)**2)/(4*K*t)) + np.exp(-((y[j]+y_o)**2)/(4*K*t))
        C_x = np.exp(-((x[i]-x_o-v*t)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t)**2)/(4*K*t))
        for n in range(1,m+1):
            C_y = C_y + np.exp(-((y[j]-y_o - 2*n*l)**2)/(4*K*t)) + np.exp(-((y[j]+y_o + 2*n*l)**2)/(4*K*t))
            C_y = C_y + np.exp(-((y[j]-y_o + 2*n*l)**2)/(4*K*t)) + np.exp(-((y[j]+y_o - 2*n*l)**2)/(4*K*t))
            C_x = C_x + np.exp(-((x[i]-x_o -v*t - 2*n*b)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t + 2*n*b)**2)/(4*K*t))
            C_x = C_x + np.exp(-((x[i]-x_o -v*t + 2*n*b)**2)/(4*K*t)) + np.exp(-((x[i]+x_o+v*t - 2*n*b)**2)/(4*K*t))
        I = 1/(4*np.pi*K*t) * C_y * C_x * np.exp(-Q*t)
        C[j][i] = convolve(S,I,mode='valid')*delta_t
        C[len(y)-1-j][i] = C[j][i]


txt_name = "./data/indoors-C-contour-Q" +str(Q) + "-t" + str(hour) + ".txt"
myfile = open(txt_name,"w")
myfile = open(txt_name,"a")
for i in range(len(x)):
    for j in range(len(y)):
        words = str(C[j][i]) + ","
        myfile.write(words)
myfile.close()

#plot
fig,ax = plt.subplots(1,1)
cp = ax.contour(X,Y,C,colors='black',levels = 12)
plt.axis('square')
plt.clabel(cp,fontsize=12, fmt='%1.0f $R$',colors='black')
ax.set_xlim([0,8])
ax.set_ylim([0,8])
ax.tick_params(axis='both',labelsize=12)
filename = "./plots/indoors-C-contour-Q" + str(Q) + "-t" + str(hour) + ".png"
plt.savefig(filename)
plt.show()