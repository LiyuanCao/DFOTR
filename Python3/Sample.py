"""sample management"""
import numpy as np

class Sample:
    def __init__(self, x, options):
        # input dimension
        self.n = len(x)

        # initial sample
        Y = x + options['tr_delta'] * np.eye(self.n)
        self.Y = np.concatenate((Y, x.reshape(1,self.n)))

        # function value of initial sample
        self.fY = np.empty(len(self.Y))
        self.fY[:] = np.nan

    @property
    def m(self):
        m = len(self.Y)
        assert m == len(self.fY), \
            'Number of sample points and number of function values do not match!'
        return m

    def Ycentered(self, center):
        return self.Y - center

    def distance(self, center):
        return np.linalg.norm(self.Ycentered(center), axis=1)

    def addpoint(self, point):
        # assert self.m < options['sample_max'],
        #     'Attemp to add more points when the sample set is already at its capacity. '
        self.Y = np.vstack((self.Y, point))
        self.fY = np.append(self.fY, [np.nan])
    
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
        self.fY = np.delete(self.fY, toremove)


def quadratic_terms(x):
    n = len(x)
    x = x[np.newaxis] # turn x from 1D to a 2D array
    temp = x * x.T - (1 - 1/np.sqrt(2)) * np.diag(pow(x, 2)[0]) 
    
    return temp[np.tril_indices(n)]