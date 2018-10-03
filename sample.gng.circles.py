import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import growingneuralgas as gng

# Load Data
data = np.loadtxt('data.circles.txt')

# Neural Gas Parameters
params = structure()
params.N = 40
params.maxit = 50
params.L = 40
params.epsilon_b = 0.2
params.epsilon_n = 0.01
params.alpha = 0.5
params.delta = 0.995
params.T = 50

# Fit Neural Gas to Data
print("Fitting Growing Neural Gas Network ...")
net = gng.fit(data, params)

plt.figure()
plt.grid()
plt.scatter(data[:,0], data[:,1], s=2)
for i in range(0, params.N):
    for j in range(i+1, params.N):
        if net.C[i,j] == 1:
            plt.plot([net.w[i,0], net.w[j,0]], [net.w[i,1], net.w[j,1]], c='r')

plt.scatter(net.w[:,0], net.w[:,1], s=60, c='y', edgecolors='r')

plt.title('GNG for circles dataset')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()
