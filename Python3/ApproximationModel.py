"""Contains a functin that build quadratic model based on the sample.
"""

import numpy as np
import scipy
from .trust_sub import *

class ApproximationModel:
    def __init__(self, n, options):
        self.n = n

        self.type = {
            'model': options['alg_model'],
            'TR': options['alg_TR'],
            'TRsub': options['alg_TRsub']
        }

        if self.type['model'] is 'linear':
            self.c = np.empty(1)
            self.g = np.empty(n)
        elif self.type['model'] is 'quadratic':
            self.c = np.empty(1)
            self.g = np.empty(n)
            self.H = np.empty((n,n))

        # if self.type['TR'] is 'ball':
        self.center = np.empty(n)
        self.delta = np.empty(1)
        
    def fit(self, samp):
        assert samp.n == self.n, 'Dimensions of model and sample mismatch.'
        m, n = samp.m, samp.n
        Ycentered = samp.Ycentered(self.center)

        if (self.type['model'] is 'quadratic') and (m > n + 1): 
            # construct the big matrix
            A = np.empty((1+n+m, 1+n+m))
            A[:1+n, :1+n] = 0
            A[0, 1+n:] = 1
            A[1+n:, 0] = 1
            A[1:1+n, 1+n:] = Ycentered.T 
            A[1+n:, 1:1+n] = Ycentered 
            A[1+n:, 1+n:] = np.dot(Ycentered, Ycentered.T) ** 2 / 2

            # the right-hand side
            b = np.zeros(1+n+m) 
            b[-m:] = samp.fY

            # solve the linear system and retrieve the gradient and the hessian 
            lamda = np.linalg.solve(A, b)
            self.c = lamda[0]
            self.g = lamda[1:1+n]
            self.H = np.dot(Ycentered.T * lamda[-m:], Ycentered)

        elif (self.type['model'] is 'quadratic') and (m == n + 1):
            # fit an well-determined linear model 
            self.H = 0

            # solve the linear system
            A = np.empty((1+n, 1+n))
            A[:,0] = 1
            A[:,1:] = Ycentered
            temp = np.linalg.solve(A, samp.fY)
            self.c = temp[0]
            self.g = temp[1:]

    def minimize(self):
        assert np.isreal(self.g).all(), 'The gradient is not all real. '

        if self.type['model'] is 'linear' or (self.type['model'] is 'quadratic' \
             and (np.isnan(self.H).any() or np.linalg.norm(self.H) < 1e-12)):
            s = - self.delta / np.linalg.norm(self.g) * self.g
            val = sum(self.g * s)

        elif (self.type['model'] is 'quadratic') and (self.type['TRsub'] is 'exact'):
            s, val = trust_sub_exact(self.H, self.g, self.delta)

        elif (self.type['model'] is 'quadratic') and (self.type['TRsub'] is 'CG'):
            s, val = trust_sub_CG(self.H, self.g, self.delta)
            
        else: 
            raise('Something went wrong!')

        return self.center + s, -val

    def update_delta(self, rho, stepSize2delta, options):
        # Updating iterate and trust-region radius.

        # When the approximation is very good, increase TR radius
        if rho >= options['tr_toexpand'] and stepSize2delta >= 0.99:
            self.delta *= options['tr_expand']
        elif (0 <= rho < options['tr_toaccept']) or rho < options['tr_toshrink']:
            # When the approximation is bad, but not complete bullshit, 
            # reduce the TR radius. 
            self.delta *= options['tr_shrink'] 




