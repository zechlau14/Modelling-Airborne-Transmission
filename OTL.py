import numpy as np
import csv
import matplotlib.pyplot as plt

#time axis
t = np.arange(180)
prob = np.zeros_like(t)
sd = 1 #social distancing limit

#mesh
l= 8
b= 8
x_o = 4
y_o = 4
delta_x = 0.05 
n_x = int(l / delta_x) + 1
x = np.linspace(0,8,n_x) 
y = np.linspace(0,8,n_x) 
X,Y = np.meshgrid(x,y)
T = np.zeros_like(X)

#file-read of TTI contour data
z = []
name = "indoors-TTI-Q0.00167-R0.5"
txt_name = "./data/" + name + ".txt"
f = open(txt_name,'r')
z = f.read().split(',')
f.close()

N_room = 0
for i in range(len(x)):
    for j in range(len(y)):
        T[j][i] = z[len(y) * i + j]
        if ((y[i]-y_o)**2 + (x[j]-x_o)**2) > sd**2: N_room = N_room + 1

# check for N_inf
N_inf = np.zeros_like(t)
for k in range(len(t)):
    for i in range(len(x)):
        for j in range(len(y)):
            if ((y[i]-y_o)**2 + (x[j]-x_o)**2) > sd**2: 
                if T[j][i] < t[k]: N_inf[k] = N_inf[k] + 1
    prob[k] = N_inf[k] / N_room * 100

if prob[-1] < 5:
    print("There is no significant risk of airborne transmission for this case!")
else:
    n = 0
    while prob[n] < 5: n = n+1
    print("Significant risk (>5%) of airborne transmission begins at " + str(int(t[n])) + " minutes!")

n = len(t) - 1
if prob[n] == 100: 
    while prob[n] == 100: n = n-1

plt.plot(t[:n],prob[:n],color='black')
plt.hlines(5,t[0],t[n],color='red',linestyle='dashed')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
filename = "./plots/prob-" + name[12:] + ".png"
plt.savefig(filename)
plt.show()
