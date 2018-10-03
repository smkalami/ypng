import numpy as np
from ypstruct import structure

# Fit a Growing Neural Gas Network
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
    L = params.L
    epsilon_b = params.epsilon_b
    epsilon_n = params.epsilon_n
    alpha = params.alpha
    delta = params.delta
    T = params.T
    
    # Initialization
    w = np.zeros((N,ndim))
    E = np.zeros(N)
    C = np.zeros((N,N))
    t = np.zeros((N,N))
    tt = 0

    K = 2
    for i in range(K):
        w[i] = np.random.uniform(xmin, xmax, (1,ndim))

    # Main Loop
    nx = 0
    for it in range(maxit):
        for l in range(ndata):
            
            # Get Input Vector
            nx += 1
            x = data[l]
            
            # Calculate Distance and Sort Neurons
            d = np.sum((x-w[0:K])**2, axis=1)    # Squared Distance
            sortorder = np.argsort(d)
            
            # Find 2 Best Neurons
            i = sortorder[0]
            j = sortorder[1]

            # Aging
            t[i,0:K] += 1
            t[0:K,i] += 1

            # Update Error
            E[i] += d[i]
            
            # Adaptation
            for k in range(K):
                if k == i:
                    eps = epsilon_b
                else:
                    if C[i,k] == 0:
                        continue
                        
                    eps = epsilon_n

                w[k] += eps*(x-w[k])

            # Create Link between 1st and 2nd Neurons
            C[i,j] = 1
            C[j,i] = 1
            t[i,j] = 0
            t[j,i] = 0

            # Remove Old Links
            old_links = t > T
            C[old_links] = 0
            nNeighbor = np.sum(C,axis=0)
            keeporder = np.argsort(-nNeighbor).tolist()
            C = C[keeporder,:][:,keeporder]
            t = t[keeporder,:][:,keeporder]
            w = w[keeporder,:]
            E = E[keeporder]
            K = np.where(nNeighbor>0)[0].size

            # Add New Nodes
            if K < N and nx % L == 0:
                q = np.argmax(E)
                f = np.argmax(C[q,:]*E)

                w[K] = (w[q]+w[f])/2
                C[K,:] = 0
                C[:,K] = 0
                t[K,:] = 0
                t[:,K] = 0

                C[q,f] = 0
                C[f,q] = 0
                C[q,K] = 1
                C[K,q] = 1
                C[f,K] = 1
                C[K,f] = 1
                
                E[q] *= alpha
                E[f] *= alpha
                E[K] = E[q]

            # Decrease Errors
            E *= delta

        # Display Iteration Info
        print("Iteration {0}".format(it))

    net = structure()
    net.w = w
    net.C = C
    net.t = t
    net.E = E

    return net
