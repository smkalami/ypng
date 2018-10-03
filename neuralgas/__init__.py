import numpy as np
from ypstruct import structure

# Fit a Neural Gas Network
def fit(data, params):

    # Data Size and Dimension
    ndata = data.shape[0]
    ndim = data.shape[1]

    # Shuffle Data Points
    np.random.shuffle(data)

    # Find Min. and Max. of Data Points
    xmin = np.amin(data, axis=0)
    xmax = np.amax(data, axis=0)

    # Parameters
    N = params.N
    maxit = params.maxit
    tmax = params.tmax
    epsilon_initial = params.epsilon_initial
    epsilon_final = params.epsilon_final
    lambda_initial = params.lambda_initial
    lambda_final = params.lambda_final
    T_initial = params.T_initial
    T_final = params.T_final

    # Initialization
    w = np.random.uniform(xmin, xmax, (N,ndim))
    C = np.zeros((N,N))
    t = np.zeros((N,N))
    tt = 0

    # Main Loop
    for it in range(maxit):
        for l in range(ndata):
            
            # Get Input Vector
            x = data[l]
            
            # Calculate Distance and Sort Neurons
            d = np.sum((x-w)**2, axis=1)    # Squared Distance
            sortorder = np.argsort(d)

            # Update Parameters
            eps = epsilon_initial * (epsilon_final/epsilon_initial)**(tt/tmax)
            lam = lambda_initial * (lambda_final/lambda_initial)**(tt/tmax)
            T = T_initial*(T_final/T_initial)**(tt/tmax)

            # Adaptation
            for ki in range(N):
                i = sortorder[ki]
                w[i] += eps*np.exp(-ki/lam)*(x-w[i])
            
            # Increment Counter
            tt += 1

            # Create Link between 1st and 2nd Neuron
            i = sortorder[0]
            j = sortorder[1]
            C[i,j] = 1
            C[j,i] = 1
            t[i,j] = 0
            t[j,i] = 0

            # Aging
            t[i,:] += 1
            t[:,i] += 1

            # Remove Old Links
            old_links = (t[i,:] > T)
            C[i, old_links] = 0
            C[old_links, i] = 0

        
        # Display Iteration Info
        print("Iteration {0}".format(it))

    net = structure()
    net.w = w
    net.C = C
    net.t = t

    return net
