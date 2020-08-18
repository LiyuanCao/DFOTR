"""sample management"""
import numpy as np
from .util import phi, alpha2cgH

class Sample:
    def __init__(self, x, bounds, options):
        # input dimension
        self.n = len(x)

        # initial sample
        Y = x + options['tr_delta'] / np.sqrt(self.n) * \
            np.random.randn(options['sample_initial']-1,self.n)
        self.Y = np.concatenate((Y, x.reshape(1,self.n)))

        # Project the sample points into the feasible region. 
        for i in range(len(self.Y)):
            for j in range(self.n):
                if bounds[j]:
                    if self.Y[i,j] < bounds[j][0]:
                        self.Y[i,j] = bounds[j][0]
                    if self.Y[i,j] > bounds[j][1]:
                        self.Y[i,j] = bounds[j][1]

        # function value of initial sample
        self.fYeval = np.empty((len(self.Y), options['alg_feval']))
        self.fYeval[:] = np.nan

    @property
    def m(self):
        m = len(self.Y)
        assert m == len(self.fYeval), \
            'Number of sample points and number of function values do not match!'
        return m
    
    @property
    def fY(self):
        return np.nanmean(self.fYeval,axis=1)

    def Ycentered(self, center):
        return self.Y - center

    def distance(self, center):
        ''' The distance of all sample points to the specified center.'''
        return np.linalg.norm(self.Ycentered(center), axis=1)

    def addpoint(self, point, model, options):
        ''' Add a point to the sample set. 
            If the sample set has already reached its capacity,
            also return the index of the point that should be removed. 
        '''
        self.Y = np.vstack((self.Y, point))
        self.fYeval = np.vstack((self.fYeval, [np.nan] * self.fYeval.shape[1]))

        if self.m > options['sample_max']:
            Phi = np.array([phi(y, model.type['model']) for y in self.Ycentered(model.center)[:-1]])
            l = np.linalg.solve(Phi.T, phi(point - model.center, model.type['model']))
            if model.type['model'] is 'linear':
                l = np.abs(l) * self.distance(model.center)[:-1]**2
            elif model.type['model'] is 'quadratic':
                l = np.abs(l) * self.distance(model.center)[:-1]**3
            return l.argmax()

    def rmpoint(self, idx):
        self.Y = np.delete(self.Y, idx, axis=0)
        self.fYeval = np.delete(self.fYeval, idx, axis=0)
    
    def auto_delete(self, model, options):
        """ Delete points that are too far away from the trust region."""
        # After the deletion, the sample size should be no less than
        #  sample_min, and no more than sample_max.

        distance = self.distance(model.center)

        # indeces of the farthest ones; 
        # We should not delete too many points so that sample size goes under sample_min. 
        farthest = np.argsort(distance)[-(self.m - options['sample_min']):]
        farthest = np.array(farthest)
        # indeces of the ones that are too far
        toofar = np.where(distance > options['sample_toremove'] * model.delta)[0]
        # indeces of the points to be removed
        if toofar.shape[0] == 0 and self.m > options['sample_max']:
            # We need to keep the sample size under its maximum. 
            toremove = farthest[-(self.m - options['sample_max']):]
        else:
            toremove = np.intersect1d(farthest, toofar)

        # remove points
        self.Y = np.delete(self.Y, toremove, axis=0)
        self.fYeval = np.delete(self.fYeval, toremove, axis=0)

    def poise(self, model, keep=[]):
        ''' Find a new point within the trust region to improve the poisedness 
        of the sample set. If the sample set size has reached its maximum, also 
        return the index of the point to be replace. 
        '''
        # sample points in the basis
        Phi = np.array([phi(y, model.type['model']) for y in self.Ycentered(model.center)])

        if (model.type['model'] is 'quadratic') and (self.m < (self.n+1) * (self.n+2) // 2):
            Q, _ = np.linalg.qr(Phi.T, mode='complete')
            # print('\n'.join([' '.join(['{:4.4f}'.format(item) for item in row]) for row in Q]))

            model.c, model.g, model.H = alpha2cgH(Q[:,-1])
            xmin, valmin = model.minimize()
            model.c, model.g, model.H = -model.c, -model.g, -model.H
            xmax, valmax = model.minimize()
            if -valmax - model.c > model.c - valmin:
                return xmax, None
            else:
                return xmin, None

        elif (model.type['model'] is 'quadratic') and (self.m == (self.n+1) * (self.n+2) // 2):
            
            # the indeces of cadidates for replacing
            candidates = [i for i in range(self.m) if i not in keep]

            # Alpha, the matrix containing the coefficeints of the Lagrange polynomials 
            A = np.linalg.inv(Phi)

            # Preallocate space for all the points. 
            xnew = np.empty((len(candidates), 2, self.n))
            fxnew = np.empty((len(candidates), 2))
            
            for i in candidates:
                model.c, model.g, model.H = alpha2cgH(A[:,i])
                xnew[i,0,:], val = model.minimize()
                fxnew[i,0] = model.c - val
                model.c, model.g, model.H = -model.c, -model.g, -model.H
                xnew[i,1,:], val = model.minimize()
                fxnew[i,1] = -val - model.c
            
            l = np.max(fxnew, axis=1)
            assert np.all(l >= 0)
            if model.type['model'] is 'linear':
                l = l * self.distance(model.center)[candidates]**2
            elif model.type['model'] is 'quadratic':
                l = l * self.distance(model.center)[candidates]**3

            idx = l.argmax()
            if fxnew[idx,0] < fxnew[idx,1]:
                return xnew[idx,1], idx
            else:
                return xnew[idx,0], idx