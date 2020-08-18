import numpy as np 

def phi(x, basis = 'quadratic'):
    n = len(x)

    if basis is 'linear':
        phix = np.empty(n+1)
        phix[0] = 1
        phix[1:] = x
    
    elif basis is 'quadratic':
        x = x[np.newaxis] # turn x from 1D to a 2D array
        phix = np.empty((n+1) * (n+2) // 2)
        phix[0] = 1
        phix[1:n+1] = x[0]
        temp = x * x.T + (1/np.sqrt(2) - 1) * np.diag(pow(x, 2)[0]) 
        phix[n+1:] = temp[np.tril_indices(n)]

    return phix
    
def alpha2cgH(alpha):
    n = int(np.round((np.sqrt(1 + 8 * len(alpha)) - 3) / 2))
    c = alpha[0]
    g = alpha[1:n+1]
    H = np.zeros((n, n))
    H[np.tril_indices(n)] = alpha[n+1:]
    H = H + H.T
    H = H + (np.sqrt(1/2) - 1) * np.diag(np.diag(H))
    return c, g, H

def qr(A):
    Q = np.empty(A.shape)
    R = np.zeros((A.shape[1], A.shape[1]))
    k = 0
    for i in range(A.shape[1]):
        projection = np.dot(Q[:,:i-k].T, A[:,i])
        residual = A[:,i] - np.dot(Q[:,:i-k], projection)
        normr = sum(residual**2)**0.5
        R[:i-k, i] = projection
        if normr == 0:
            k += 1
        else: 
            Q[:,i-k] = residual / normr
            R[i-k,i] = normr
    if k > 0:
        Q = Q[:,:-k]
        R = R[:-k,:]
    return Q, R

if __name__ == '__main__':
    x = np.random.randint(0,10, 5)
    print(x)

    print(phi(x))