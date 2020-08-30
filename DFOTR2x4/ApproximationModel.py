"""Contains a functin that build quadratic model based on the sample.
"""

import numpy as np
import scipy
from scipy import optimize
from .trust_sub import *
from .util import phi, qr, alpha2cgH

class ApproximationModel:
    def __init__(self, n, bounds, options):
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

        self.bounds = bounds
        self.center = np.empty(n)
        self.delta = np.empty(1)
        
    def fit(self, samp):
        assert samp.n == self.n, 'Dimensions of model and sample mismatch.'
        m, n = samp.m, samp.n
        Ycentered = samp.Ycentered(self.center)

        # scale = 0.1 / np.mean(np.abs(Ycentered), axis=0)
        # Ycentered = scale * Ycentered 

        # Phi = np.array([phi(x, basis = self.type['model']) for x in Ycentered])
        # Q, R = qr(Phi.T)
        # coeff = np.linalg.solve(np.dot(R, R.T), np.dot(R, samp.fY))   
        # coeff = np.dot(Q, coeff)

        # if self.type['model'] is 'linear':
        #     self.c = coeff[0]
        #     self.g = coeff[1:]
        # elif self.type['model'] is 'quadratic':
        #     self.c, self.g, self.H = alpha2cgH(coeff)
        #     self.g = self.g * scale
        #     self.H = self.H * scale
        #     self.H = self.H.T * scale

        # temp = np.empty((samp.m, 3))
        # temp[:,0] = [self(x) for x in samp.Y]
        # temp[:,1] = samp.fY
        # temp[:,2] = np.dot(Phi, coeff)
        # print(temp)
        # a = 1

        if m < n + 1:
            ''' minimum norm interpolation '''
            Phi = np.array([phi(y, basis=self.type['model']) for y in samp.Ycentered(self.center)])
            alpha = np.dot(Phi.T, np.linalg.solve(np.dot(Phi, Phi.T), samp.fY))
            if self.type['model'] is 'linear':
                self.c, self.g = alpha[0], alpha[1:]
            if self.type['model'] is 'quadratic':
                self.c, self.g, self.H = alpha2cgH(alpha)
        elif (m == n + 1) and (self.type['model'] is  'linear'):
            ''' linear interpolation '''
            A = np.empty((1+n, 1+n))
            A[:,0] = 1
            A[:,1:] = Ycentered
            temp = np.linalg.solve(A, samp.fY)
            self.c = temp[0]
            self.g = temp[1:]
        elif (self.type['model'] is 'quadratic') and (m >= n + 1) and (m <= (n+1)*(n+2)/2): 
            try:
                ''' minimum Frobenius interpolation '''
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
            except np.linalg.LinAlgError as err:
                if 'Singular matrix' in str(err):
                    # Use minimum norm interpolation if minimum Frobenius 
                    # interpolation lead to singularity. 
                    Phi = np.array([phi(y) for y in samp.Ycentered(self.center)])
                    alpha = np.dot(Phi.T, np.linalg.solve(np.dot(Phi, Phi.T), samp.fY))
                    if self.type['model'] is 'linear':
                        self.c, self.g = alpha[0], alpha[1:]
                    if self.type['model'] is 'quadratic':
                        self.c, self.g, self.H = alpha2cgH(alpha)
                
                else: 
                    raise


    def __call__(self, x):
        x = x - self.center
        if self.type['model'] is 'linear':
            return self.c + sum(self.g * x)
        elif self.type['model'] is 'quadratic':
            return self.c + sum(self.g * x) + 0.5 * np.dot(x, np.dot(self.H, x))

    def minimize(self):
        ''' Minimize the approximation model. 
            Return the minimizer and the minimum value.
        '''
        assert np.isreal(self.g).all(), 'The gradient is not all real. '

        # Solve the trust region subproblem. 
        if self.type['TR'] is 'ball':
            if self.type['model'] is 'linear' or (self.type['model'] is 'quadratic' \
                and (np.isnan(self.H).any() or np.linalg.norm(self.H) < 1e-12)):
                normg = np.linalg.norm(self.g)
                if normg > 0:
                    s = - self.delta / normg * self.g
                else: 
                    s = np.zeros(self.n)
                val = sum(self.g * s)

            elif (self.type['model'] is 'quadratic') and (self.type['TRsub'] is 'exact'):
                s, val = trust_sub_exact(self.H, self.g, self.delta)

            elif (self.type['model'] is 'quadratic') and (self.type['TRsub'] is 'CG'):
                s, val = trust_sub_CG(self.H, self.g, self.delta)
                
            else: 
                raise('Something went wrong!')

            x1 = self.center + s
            val = self.c + val

        elif self.type['TR'] is 'box':
            bounds = np.array([[max(self.center[i] - self.delta, self.bounds[i][0]), 
                      min(self.center[i] + self.delta, self.bounds[i][1])]
                      for i in range(self.n)])
                      
            x0s = (bounds[:,0][:, np.newaxis] + np.random.rand(self.n, 10) * (bounds[:,1] - bounds[:,0])[:, np.newaxis]).T
            x0s = np.vstack((x0s, np.mean(bounds, axis=1)))
            mx0s = np.array([self(x) for x in x0s])
            minidx = mx0s.argmin()
            x0 = x0s[minidx]

            jac = lambda x: self.g + np.sum(self.H * (x - self.center), axis=1)

            # result = optimize.minimize(self, x0, 
            #                   method = 'TNC',
            #                   jac = jac,
            #                   hess = lambda x: self.H, 
            #                   bounds = bounds
            # )
            # result = optimize.minimize(self, x0,
            #                     method = 'trust-constr',
            #                     jac = jac,
            #                     hess = lambda x: self.H, 
            #                     bounds = bounds
            # )
            result = optimize.minimize(self, x0,
                                method = 'L-BFGS-B',
                                jac = jac,
                                hess = lambda x: self.H, 
                                bounds = bounds
            )
            x1, val = result['x'], result['fun']

        return x1, val

    def update_delta(self, rho, stepSize2delta, options):
        # Updating iterate and trust-region radius.

        # When the approximation is very good, increase TR radius
        if rho >= options['tr_toexpand'] and stepSize2delta >= 0.99:
            self.delta *= options['tr_expand']
        elif (0 <= rho < options['tr_toaccept']) or rho < options['tr_toshrink']:
            # When the approximation is bad, but not complete bullshit, 
            # reduce the TR radius. 
            self.delta *= options['tr_shrink'] 




