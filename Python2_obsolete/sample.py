"""sample management"""
import numpy as np
from .build_model import quadraticModel
from .trust_sub import trust_sub

class sample:
    def __init__(self, x, func, par):
        # input dimension
        self.n = par.n

        # initial sample size
        self.m = par.sample_initial

        # number of sample points within the acceptable range
        self.mgood = self.m

        # initial sample
        np.random.seed(0)
        Y = np.random.rand(self.m - 1, par.n) * 2 - 1
        Y = par.tr_delta / np.sqrt(par.n) * Y
        self.Y = np.concatenate((x + Y, x.reshape(1,par.n) ))

        # function value of initial sample
        self.f = np.array( map(func, self.Y) )

        # the second stage of initialization
        self.initialization2nd(x, Y, par)


    def initialization2nd(self, x, Y, par):
        ''' second stage of initialization
        The initialization is split into two part for the scenario where 
        the objective function can not be passed as a Python function.
        '''
        # centering origin
        self.center = x
        self.fcenter = self.f[-1]
        # self.centeridx = self.m

        # centered sample
        self.Ycentered = Y

        # centered function value
        self.fcentered = self.f[:-1] - self.fcenter

        # distance of each point in the sample to the current point
        self.distance = np.append(np.linalg.norm(self.Ycentered, axis=1), 0.0)

    
    def update(self, x, f, x1, f1, par, iteration):
        """ The sample is updated every iteration with this function."""

        # add new point
        self.Y = np.concatenate((self.Y, x1.reshape(1, par.n) ))
        self.f = np.append(self.f, f1)

        # update distance
        self.distance = np.linalg.norm(self.Y - x.reshape(1, par.n), axis=1)

        
        # We first delete points. After the deletion, the sample size should
        # be no less tham (par.sample_min - 1), and no more than 
        # (par.sample_max - 1). The "-1" is due to the point we are going 
        # to add into the sample later.

        # indeces of the farthest ones; 
        # We should not delete too many points so that sample size goes under par.sample_min. 
        farthest = np.argsort(self.distance)[-(self.m - par.sample_min):]
        farthest = np.array(farthest)
        # indeces of the ones that are too far
        toofar = np.where(self.distance > par.sample_toremove * par.tr_delta)[0]
        # indeces of the points to be removed
        if toofar.shape[0] == 0 and self.m > par.sample_max:
            # We need to keep the sample size under its maximum. 
            toremove = farthest[-(self.m - par.sample_max):]
        else:
            toremove = np.intersect1d(farthest, toofar)

        # remove points
        self.Y = np.delete(self.Y, toremove, axis=0)
        self.f = np.delete(self.f, toremove)

        # update sample size
        self.m = self.Y.shape[0]
        self.mgood = self.m + toremove.shape[0] - toofar.shape[0]

        # update centering origin 
        self.center = x
        self.fcenter = f
        self.centeridx = np.where((self.Y == self.center).all(axis=1))[0]

        # update centered sample
        self.Ycentered = self.Y[np.arange(self.m) != self.centeridx] - x.reshape(1, par.n)

        # update centered function value
        self.fcentered = self.f[np.arange(self.m) != int(self.centeridx)] - f1

    def addpoints(self, points, fvalues, par):
        # number of new points
        p = points.shape[0]

        # sample size 
        self.m += p

        # sample
        self.Y = np.concatenate((self.Y, points))

        # centered points 
        centeredPoints = points - self.center
        self.Ycentered = np.concatenate((self.Ycentered, centeredPoints))

        # function values
        self.f = np.concatenate((self.f, fvalues))

        # centered function values
        self.fcentered = np.concatenate((self.fcentered, fvalues - self.fcenter))

        # distance to the center
        d = np.linalg.norm(centeredPoints, axis=1)
        self.distance = np.concatenate((self.distance, d))

        # number of sample points within the acceptable range
        self.mgood += sum(d < par.tr_delta * par.sample_toremove)



def quadratic_terms(x):
    n = len(x)
    x = x[np.newaxis] # turn x from 1D to a 2D array
    temp = x * x.T - (1 - 1/np.sqrt(2)) * np.diag(pow(x, 2)[0]) 
    
    return temp[np.tril_indices(n)]