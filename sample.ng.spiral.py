import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import neuralgas as ng

# Load Data
data = np.loadtxt('data.spiral.txt')

# Neural Gas Parameters
params = structure()
params.N = 100
params.maxit = 80
params.tmax = 8000
params.epsilon_initial = 0.9
params.epsilon_final = 0.4
params.lambda_initial = 10
params.lambda_final = 1
params.T_initial = 5
params.T_final = 10

# Fit Neural Gas to Data
print("Fitting Neural Gas Network ...")
net = ng.fit(data, params)

plt.figure()
plt.grid()
plt.scatter(data[:,0], data[:,1], s=2)
for i in range(0, params.N):
    for j in range(i+1, params.N):
        if net.C[i,j] == 1:
            plt.plot([net.w[i,0], net.w[j,0]], [net.w[i,1], net.w[j,1]], c='r')

plt.scatter(net.w[:,0], net.w[:,1], s=60, c='y', edgecolors='r')

plt.title('Neural Gas for spiral dataset')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()
